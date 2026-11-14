import json
import logging

from httpx import ASGITransport, AsyncClient

from app.logging_conf import JsonFormatter, request_id_var, setup_logging
from app.main import app, get_llm, get_search
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
