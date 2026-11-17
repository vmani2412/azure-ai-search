import json
import logging

from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response

from app.logging_conf import JsonFormatter, request_id_var, setup_logging
from app.main import app, get_azure_sync, get_llm, get_qdrant, get_search
from app.mcp_server import get_commit_diff, get_deployment_timestamps, get_recent_commits, mcp
from app.services.azure_sync import AzureSyncError
from app.services.llm import (
    ContentFilterError,
    LLMClient,
    LLMResult,
    RateLimitExhaustedError,
)


class StubLLM:
    def __init__(self, outcome: LLMResult | Exception) -> None:
        self.outcome = outcome
        self.last_context: str | None = None

    async def chat(
        self,
        message: str,
        max_tokens: int = 1000,
        context: str | None = None,
    ) -> LLMResult:
        self.last_context = context
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class StubSearch:
    def __init__(self, result: str, configured: bool = True) -> None:
        self.result = result
        self.is_configured = configured

    async def search(self, query: str) -> str:
        return self.result


class StubQdrant:
    def __init__(self, search_result: str = "", count: int = 3) -> None:
        self.search_result = search_result
        self.count = count

    async def ingest_docs(self) -> int:
        return self.count

    async def search(self, query: str) -> str:
        return self.search_result


async def test_health_endpoint() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health", headers={"X-Request-ID": "req-123"})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    assert resp.headers["X-Request-ID"] == "req-123"


async def test_chat_endpoint_success() -> None:
    stub = StubLLM(LLMResult(text="hello world", prompt_tokens=4, completion_tokens=6, retries=1))
    app.dependency_overrides[get_llm] = lambda: stub  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/chat", json={"message": "hi", "max_tokens": 100})
        assert resp.status_code == 200
        assert resp.json() == {
            "answer": "hello world",
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "retries": 1,
        }
        assert "X-Request-ID" in resp.headers
    finally:
        app.dependency_overrides.clear()


async def test_chat_endpoint_with_search_context() -> None:
    stub_llm = StubLLM(
        LLMResult(text="grounded answer", prompt_tokens=10, completion_tokens=5, retries=0)
    )
    stub_search = StubSearch("[Source: doc.txt]\ngrounded fact")
    app.dependency_overrides[get_llm] = lambda: stub_llm  # type: ignore[return-value]
    app.dependency_overrides[get_search] = lambda: stub_search  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/chat", json={"message": "question"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "grounded answer"
        assert stub_llm.last_context == "[Source: doc.txt]\ngrounded fact"
    finally:
        app.dependency_overrides.clear()


async def test_qdrant_ingest_and_chat_endpoints() -> None:
    stub_llm = StubLLM(
        LLMResult(text="Nike costs 2500", prompt_tokens=20, completion_tokens=8, retries=0)
    )
    stub_qdrant = StubQdrant("[Source: products.json]\nname: Nike Running Shoes\nprice: 2500", 3)
    app.dependency_overrides[get_llm] = lambda: stub_llm  # type: ignore[return-value]
    app.dependency_overrides[get_qdrant] = lambda: stub_qdrant  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            ing = await ac.post("/qdrant/ingest")
            assert ing.status_code == 200
            assert ing.json()["points_indexed"] == 3

            chat_resp = await ac.post("/qdrant/chat", json={"message": "Nike price?"})
            assert chat_resp.status_code == 200
            assert chat_resp.json()["answer"] == "Nike costs 2500"
            assert "Nike Running Shoes" in (stub_llm.last_context or "")
    finally:
        app.dependency_overrides.clear()


async def test_azure_sync_endpoint() -> None:
    class StubAzureSync:
        async def sync_and_verify(self) -> dict[str, object]:
            return {
                "status": "ok",
                "storage_container": "product",
                "blobs_uploaded": ["products.json", "sproduct.json"],
                "search_index": "search-1794318221652",
                "documents_indexed": 33,
                "verified_search_hits": 33,
            }

    app.dependency_overrides[get_azure_sync] = lambda: StubAzureSync()  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/azure/sync")
            assert resp.status_code == 200
            assert resp.json()["documents_indexed"] == 33
    finally:
        app.dependency_overrides.clear()


async def test_azure_sync_error_returns_400() -> None:
    class FailingSync:
        async def sync_and_verify(self) -> dict[str, object]:
            raise AzureSyncError("Need admin key")

    app.dependency_overrides[get_azure_sync] = lambda: FailingSync()  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/azure/sync")
            assert resp.status_code == 400
            assert resp.json()["error"] == "azure_sync_error"
    finally:
        app.dependency_overrides.clear()


async def test_chat_rate_limited_returns_429() -> None:
    stub = StubLLM(RateLimitExhaustedError(retry_after_s=5.0))
    app.dependency_overrides[get_llm] = lambda: stub  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/chat", json={"message": "hi"})
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "5"
        assert resp.json()["error"] == "rate_limited"
    finally:
        app.dependency_overrides.clear()


async def test_chat_content_filtered_returns_422() -> None:
    stub = StubLLM(ContentFilterError({"hate": {"filtered": True}}))
    app.dependency_overrides[get_llm] = lambda: stub  # type: ignore[return-value]
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post("/chat", json={"message": "bad"})
        assert resp.status_code == 422
        assert resp.json()["error"] == "content_filtered"
    finally:
        app.dependency_overrides.clear()


def test_json_formatter_includes_request_id() -> None:
    setup_logging("INFO")
    token = request_id_var.set("trace-abc")
    try:
        record = logging.LogRecord(
            name="app.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello_event",
            args=(),
            exc_info=None,
        )
        record.ctx = {"k": "v"}
        formatted = JsonFormatter().format(record)
        data = json.loads(formatted)
        assert data["msg"] == "hello_event"
        assert data["request_id"] == "trace-abc"
        assert data["ctx"] == {"k": "v"}
    finally:
        request_id_var.reset(token)
    assert LLMClient is not None


async def test_mcp_git_and_deployment_tools() -> None:
    commits = get_recent_commits(limit=3)
    assert len(commits) >= 1
    assert "sha" in commits[0]
    assert "timestamp" in commits[0]

    diff_stat = get_commit_diff(target="HEAD", stat_only=True)
    assert diff_stat["target"] == "HEAD"
    assert diff_stat["stat"] != ""

    diff_full = get_commit_diff(target="HEAD", stat_only=False)
    assert "diff" in diff_full

    def gh_handler(_: Request) -> Response:
        return Response(
            200,
            json={
                "workflow_runs": [
                    {
                        "id": 101,
                        "name": "CI/CD Pipeline",
                        "head_branch": "main",
                        "head_sha": "b5cf119abcdef",
                        "status": "completed",
                        "conclusion": "success",
                        "run_started_at": "2026-11-14T10:00:00Z",
                        "updated_at": "2026-11-14T10:02:00Z",
                        "html_url": "https://github.com/vmani2412/azure-ai-search/actions/runs/101",
                    }
                ]
            },
        )

    async with AsyncClient(transport=MockTransport(gh_handler)) as gh_client:
        deployments = await get_deployment_timestamps(limit=2, http_client=gh_client)
        assert deployments["latest_deployment_timestamp"] == "2026-11-14T10:02:00Z"
        assert len(deployments["ci_cd_workflow_runs"]) == 1

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        c_resp = await ac.get("/mcp/commits?limit=2")
        assert c_resp.status_code == 200
        assert len(c_resp.json()["commits"]) >= 1

        d_resp = await ac.get("/mcp/diff?target=HEAD&stat_only=true")
        assert d_resp.status_code == 200
        assert d_resp.json()["target"] == "HEAD"

    assert mcp is not None
