# Project Specification Prompt: Production Azure AI RAG, Qdrant & FastMCP Service

Build a production-grade Python 3.11+ FastAPI microservice (`azure-ai-service`) with **Azure OpenAI**, **Azure AI Search**, **Azure Blob Storage sync**, a **Qdrant Vector Database** RAG pipeline, a **Git / CI-CD FastMCP Server**, **Docker Compose**, and a **GitHub Actions CI/CD pipeline** to **Azure Container Apps**.

---

## 1. Repository Structure

Create the following layout:

```text
azure-ai-search/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI (ruff, mypy, pytest) + CD (Azure Container Apps)
├── docs/
│   └── products.json               # Sample JSON data for Qdrant & Azure Storage sync
├── scripts/
│   └── loadtest_429.py             # Concurrency & 429 retry benchmark script
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app, lifespan, middleware, endpoints, error handlers
│       ├── config.py               # pydantic-settings env-based config
│       ├── schemas.py              # Pydantic request/response models
│       ├── logging_conf.py         # Structured JSON logging with contextvar request_id
│       ├── mcp_server.py           # FastMCP Git & CI/CD inspector server
│       └── services/
│           ├── __init__.py
│           ├── llm.py              # Azure OpenAI wrapper (429 retry, content filter, grounding)
│           ├── search.py           # Azure AI Search REST client
│           ├── qdrant.py           # Qdrant vector ingestion & search for local docs/
│           └── azure_sync.py       # Sync docs/ to Azure Blob Storage, index in Azure AI Search & verify
├── tests/
│   ├── test_llm.py                 # Unit tests for LLM, Search, Qdrant, AzureSync
│   └── test_main.py                # API endpoint, logging & MCP unit tests
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 2. Core Requirements & Architecture

### A. Configuration & Observability
- **`src/app/config.py`**: Use `pydantic-settings` (`BaseSettings`) with `env_file=".env"`, `extra="ignore"`, and `SecretStr` for all secrets:
  - Azure OpenAI: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`
  - Azure AI Search: `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_API_KEY`, `AZURE_SEARCH_INDEX`, `AZURE_SEARCH_TOP_K`
  - Azure Blob Storage: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_CONTAINER`
  - Qdrant: `QDRANT_URL` (default `http://localhost:6333`), `QDRANT_API_KEY`, `QDRANT_COLLECTION` (default `docs_collection`), `QDRANT_TOP_K`, `DOCS_DIR` (default `docs`)
  - Retry & Logging: `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `LLM_BASE_DELAY_S`, `LLM_MAX_DELAY_S`, `LOG_LEVEL`
- **`src/app/logging_conf.py`**: Emit structured JSON logs including `request_id` via `contextvars.ContextVar`.
- **Middleware**: Attach/propagate `X-Request-ID` on every HTTP request and response.

### B. Azure OpenAI Client (`src/app/services/llm.py`)
- Wrap `AsyncAzureOpenAI` with `max_retries=0` so retry behavior is explicitly controlled and measurable.
- Use `max_completion_tokens` (compatible with both `gpt-4o-mini` and reasoning models like `gpt-5-mini`; default `max_tokens=1000`).
- **429 Retry Policy**:
  - Honor `retry-after-ms` and `retry-after` response headers when present.
  - Otherwise use exponential backoff with full jitter: `uniform(0, min(cap, base * 2**attempt))`.
  - After `llm_max_retries`, raise `RateLimitExhaustedError` → HTTP `429` with `Retry-After` header.
- **Content Filter Handling**:
  - Detect input/output Azure `content_filter` errors, log only categories/severities (never user text), and raise `ContentFilterError` → HTTP `422`.
- **Strict RAG Grounding**:
  - Accept optional `context: str | None = None`.
  - If `context` is provided and empty/whitespace, return `"Requested data is not available."` immediately with `0` tokens used (no LLM call).
  - If `context` is non-empty, send a strict system prompt instructing the model to answer **only** from `PROVIDED CONTEXT` or reply `"Requested data is not available."`.

### C. Azure AI Search Grounded RAG (`src/app/services/search.py`)
- Query Azure AI Search REST API (`POST {endpoint}/indexes/{index}/docs/search?api-version=2024-07-01`) via `httpx.AsyncClient`.
- Extract both unstructured text fields (`content`, `chunk`, `merged_content`, `text`) and structured JSON fields (`product_id`, `name`, `category`, `price`, `description`, etc.), ignoring `@search.*`, `metadata_*`, and `AzureSearch_DocumentKey`.
- Wire into `POST /chat`: if Azure AI Search is configured, search first and pass the retrieved context to `LLMClient.chat()`.

### D. Azure Blob Storage → Azure AI Search Sync API (`src/app/services/azure_sync.py`)
- Expose `POST /azure/sync` (`AzureSyncResponse`):
  1. Upload all files in `docs/` to Azure Blob Storage using SharedKey HMAC-SHA256 REST signing (`2021-08-06`), creating the container if needed.
  2. Check/create the Azure AI Search index (`GET/PUT /indexes/{index}`), returning a clear HTTP `400` error if a read-only Query key is used instead of a Primary Admin key.
  3. Unwrap JSON arrays or single-list wrapper objects (e.g. `{"products": [...]}`) and push all records via `mergeOrUpload` (`POST /indexes/{index}/docs/index`).
  4. Verify indexed documents via `POST /indexes/{index}/docs/search` (`"count": true`) and return `blobs_uploaded`, `documents_indexed`, and `verified_search_hits`.

### E. Qdrant Vector Database API (`src/app/services/qdrant.py`)
- Read all `.json`, `.txt`, `.md`, and `.csv` files from `docs/` (unwrapping JSON lists and `{"products": [...]}` wrappers).
- Compute unit-normalized 256-dim dense vectors (word + character n-gram feature hashing) and store points in Qdrant via REST API (`DELETE` + `PUT /collections/{collection}` for idempotent re-ingestion, then `PUT /collections/{collection}/points?wait=true`).
- Expose two separate endpoints:
  - `POST /qdrant/ingest`: Ingests all files from `docs/` into Qdrant and returns `points_indexed`.
  - `POST /qdrant/chat`: Searches Qdrant (`POST /collections/{collection}/points/search`, auto-ingesting on first call if needed) and passes matched chunks to `LLMClient.chat()`.

### F. Git / CI-CD FastMCP Server (`src/app/mcp_server.py`)
- Build an MCP server using `fastmcp` (`FastMCP("git-cicd-inspector")`) exposing 3 tools with JSON-serializable parameters:
  1. `get_recent_commits(limit: int = 10, branch: str = "HEAD")`
  2. `get_commit_diff(target: str = "HEAD", stat_only: bool = False)`
  3. `get_deployment_timestamps(limit: int = 5)` — combines local Git commit timestamps with GitHub Actions workflow run timestamps via GitHub REST API.
- Support both `MCP_TRANSPORT=http` (default port `8001` at `/mcp`) and `MCP_TRANSPORT=stdio`.
- Also expose REST wrappers on FastAPI: `GET /mcp/commits`, `GET /mcp/diff`, `GET /mcp/deployments`.

---

## 3. Infrastructure, Docker & CI/CD

- **`Dockerfile`**: Python `3.11-slim`, install `git`, create non-root `appuser`, mark `/app` as git safe directory, copy `pyproject.toml`, `src/`, and `docs/`, run `pip install --no-cache-dir .`, expose `8000`, run `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- **`docker-compose.yml`**: Define 3 services:
  1. `qdrant` (`qdrant/qdrant:latest` on port `6333`)
  2. `azure-ai-service` (port `8000`, `QDRANT_URL=http://qdrant:6333`, mounts `./docs:/app/docs:ro` and `./.git:/app/.git:ro`)
  3. `mcp-server` (reuses built image, runs `python -m app.mcp_server` on port `8001`, mounts `./.git:/app/.git:ro`)
- **`.github/workflows/ci.yml`**:
  - `checks` job (PRs & `main`): install `.[dev]`, run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest`.
  - `deploy` job (push to `main` after `checks` passes): log in to Azure & ACR, build and deploy container image to Azure Container Apps (`targetPort: 8000`) using `azure/container-apps-deploy-action@v2`.

---

## 4. Quality Gates & Testing Requirements

- Configure `pyproject.toml` with:
  - `ruff` (`line-length = 100`, rules `["E", "F", "I", "B", "UP", "ASYNC"]`)
  - `mypy` (`strict = true`, `python_version = "3.11"`)
  - `pytest` (`asyncio_mode = "auto"`, `--cov=app --cov-fail-under=80`)
- All unit tests must mock external HTTP/SDK calls (`httpx.MockTransport` and dependency overrides) and pass **without** a `.env` file present (so GitHub Actions CI passes cleanly).
- Verify all checks pass before finishing:
  ```bash
  ruff check . && ruff format --check . && mypy src && pytest
  ```
