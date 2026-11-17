import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_conf import request_id_var, setup_logging
from app.mcp_server import get_commit_diff, get_deployment_timestamps, get_recent_commits
from app.schemas import AzureSyncResponse, ChatRequest, ChatResponse, IngestResponse
from app.services.azure_sync import AzureSyncError, AzureSyncService
from app.services.llm import ContentFilterError, LLMClient, RateLimitExhaustedError
from app.services.qdrant import QdrantService
from app.services.search import SearchClient

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.llm = LLMClient(settings)  # one client per process, reused across requests
    app.state.search = SearchClient(settings)
    app.state.qdrant = QdrantService(settings)
    app.state.azure_sync = AzureSyncService(settings)
    yield


app = FastAPI(title="Azure AI service", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        log.info(
            "http_request",
            extra={
                "ctx": {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                }
            },
        )
        return response
    finally:
        request_id_var.reset(token)


def get_llm(request: Request) -> LLMClient:
    client: LLMClient = request.app.state.llm
    return client


def get_search(request: Request) -> SearchClient | None:
    search: SearchClient | None = getattr(request.app.state, "search", None)
    return search


def get_qdrant(request: Request) -> QdrantService:
    qdrant: QdrantService = request.app.state.qdrant
    return qdrant


def get_azure_sync(request: Request) -> AzureSyncService:
    sync: AzureSyncService = request.app.state.azure_sync
    return sync


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    search: Annotated[SearchClient | None, Depends(get_search)] = None,
) -> ChatResponse:
    context: str | None = None
    if search is not None and search.is_configured:
        context = await search.search(req.message)
    result = await llm.chat(req.message, req.max_tokens, context=context)
    return ChatResponse(
        answer=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retries=result.retries,
    )


@app.post("/qdrant/ingest", response_model=IngestResponse)
async def qdrant_ingest(
    qdrant: Annotated[QdrantService, Depends(get_qdrant)],
) -> IngestResponse:
    count = await qdrant.ingest_docs()
    return IngestResponse(
        status="ok",
        collection=get_settings().qdrant_collection,
        points_indexed=count,
    )


@app.post("/qdrant/chat", response_model=ChatResponse)
async def qdrant_chat(
    req: ChatRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    qdrant: Annotated[QdrantService, Depends(get_qdrant)],
) -> ChatResponse:
    context = await qdrant.search(req.message)
    result = await llm.chat(req.message, req.max_tokens, context=context)
    return ChatResponse(
        answer=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retries=result.retries,
    )


@app.post("/azure/sync", response_model=AzureSyncResponse)
async def azure_sync_docs(
    azure_sync: Annotated[AzureSyncService, Depends(get_azure_sync)],
) -> AzureSyncResponse:
    res = await azure_sync.sync_and_verify()
    return AzureSyncResponse(**res)


@app.get("/mcp/commits")
async def mcp_commits(limit: int = 10, branch: str = "HEAD") -> dict[str, object]:
    return {"commits": get_recent_commits(limit=limit, branch=branch)}


@app.get("/mcp/diff")
async def mcp_diff(target: str = "HEAD", stat_only: bool = False) -> dict[str, object]:
    return get_commit_diff(target=target, stat_only=stat_only)


@app.get("/mcp/deployments")
async def mcp_deployments(limit: int = 5) -> dict[str, object]:
    return await get_deployment_timestamps(limit=limit)


@app.exception_handler(AzureSyncError)
async def azure_sync_failed(_: Request, exc: AzureSyncError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "azure_sync_error", "detail": str(exc)},
    )


@app.exception_handler(RateLimitExhaustedError)
async def rate_limited(_: Request, exc: RateLimitExhaustedError) -> JSONResponse:
    headers = {"Retry-After": str(int(exc.retry_after_s))} if exc.retry_after_s else {}
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": "Upstream model is busy, retry later."},
        headers=headers,
    )


@app.exception_handler(ContentFilterError)
async def filtered(_: Request, exc: ContentFilterError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "content_filtered", "detail": "Request blocked by content policy."},
    )
