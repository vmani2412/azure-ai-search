import random
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.azure_sync import AzureSyncError, AzureSyncService
from app.services.llm import (
    NO_DATA_MESSAGE,
    ContentFilterError,
    LLMClient,
    RateLimitExhaustedError,
    backoff_delay,
)
from app.services.qdrant import QdrantService, load_docs_from_folder
from app.services.search import SearchClient


def make_settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "azure_openai_endpoint": "https://x.openai.azure.com",
        "azure_openai_api_key": SecretStr("k"),
        "azure_openai_deployment": "d",
        "azure_search_endpoint": None,
        "azure_search_api_key": None,
        "llm_max_retries": 3,
        "llm_base_delay_s": 1.0,
        "llm_max_delay_s": 8.0,
    }
    base.update(kw)
    return Settings(**base)


def err_429(
    retry_after: str | None = None, retry_after_ms: str | None = None
) -> openai.RateLimitError:
    headers: dict[str, str] = {}
    if retry_after:
        headers["retry-after"] = retry_after
    if retry_after_ms:
        headers["retry-after-ms"] = retry_after_ms
    req = httpx.Request("POST", "https://x")
    return openai.RateLimitError(
        "429", response=httpx.Response(429, headers=headers, request=req), body=None
    )


def err_filter() -> openai.BadRequestError:
    req = httpx.Request("POST", "https://x")
    body = {
        "code": "content_filter",
        "innererror": {
            "content_filter_result": {"violence": {"filtered": True, "severity": "high"}}
        },
    }
    return openai.BadRequestError("400", response=httpx.Response(400, request=req), body=body)


class Msg:
    content = "hi"


class Choice:
    def __init__(self, finish: str = "stop") -> None:
        self.message = Msg()
        self.finish_reason = finish


class Usage:
    prompt_tokens = 5
    completion_tokens = 7


class Resp:
    def __init__(self, finish: str = "stop") -> None:
        self.choices = [Choice(finish)]
        self.usage = Usage()


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **_: Any) -> Any:
        self.calls += 1
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, s: float) -> None:
        self.delays.append(s)


async def test_success_no_retries() -> None:
    c = LLMClient(make_settings(), FakeCompletions([Resp()]), SleepRecorder())
    r = await c.chat("hello")
    assert (r.text, r.retries, r.prompt_tokens) == ("hi", 0, 5)


async def test_honors_retry_after_header() -> None:
    sleep = SleepRecorder()
    fake = FakeCompletions([err_429("7"), Resp()])
    r = await LLMClient(make_settings(), fake, sleep).chat("hello")
    assert sleep.delays == [7.0]
    assert r.retries == 1


async def test_honors_retry_after_ms_header() -> None:
    sleep = SleepRecorder()
    fake = FakeCompletions([err_429(retry_after_ms="1500"), Resp()])
    r = await LLMClient(make_settings(), fake, sleep).chat("hello")
    assert sleep.delays == [1.5]
    assert r.retries == 1


async def test_backoff_when_no_header() -> None:
    sleep = SleepRecorder()
    fake = FakeCompletions([err_429(), err_429(), Resp()])
    c = LLMClient(make_settings(), fake, sleep, rng=random.Random(1))
    r = await c.chat("hello")
    assert r.retries == 2
    assert 0 <= sleep.delays[0] <= 1.0
    assert 0 <= sleep.delays[1] <= 2.0


async def test_gives_up_after_max_retries() -> None:
    fake = FakeCompletions([err_429("1")] * 4)
    with pytest.raises(RateLimitExhaustedError) as ei:
        await LLMClient(make_settings(), fake, SleepRecorder()).chat("hello")
    assert fake.calls == 4  # 1 try + 3 retries
    assert ei.value.retry_after_s == 1.0


async def test_content_filter_input_maps_to_domain_error() -> None:
    fake = FakeCompletions([err_filter()])
    with pytest.raises(ContentFilterError) as ei:
        await LLMClient(make_settings(), fake, SleepRecorder()).chat("bad")
    assert ei.value.categories["violence"]["severity"] == "high"
    assert fake.calls == 1  # never retried


async def test_content_filter_on_output() -> None:
    fake = FakeCompletions([Resp("content_filter")])
    with pytest.raises(ContentFilterError):
        await LLMClient(make_settings(), fake, SleepRecorder()).chat("x")


async def test_other_bad_request_propagates() -> None:
    req = httpx.Request("POST", "https://x")
    other = openai.BadRequestError(
        "400", response=httpx.Response(400, request=req), body={"code": "x"}
    )
    with pytest.raises(openai.BadRequestError):
        await LLMClient(make_settings(), FakeCompletions([other]), SleepRecorder()).chat("x")


def test_backoff_is_capped() -> None:
    rng = random.Random(0)
    assert all(backoff_delay(20, 1.0, 8.0, rng) <= 8.0 for _ in range(50))


async def test_grounded_empty_context_returns_no_data_without_calling_llm() -> None:
    fake = FakeCompletions([Resp()])
    c = LLMClient(make_settings(), fake, SleepRecorder())
    r = await c.chat("unknown question", context="   ")
    assert r.text == NO_DATA_MESSAGE
    assert r.prompt_tokens == 0
    assert fake.calls == 0


async def test_grounded_with_context_calls_llm() -> None:
    fake = FakeCompletions([Resp()])
    c = LLMClient(make_settings(), fake, SleepRecorder())
    r = await c.chat("question", context="Fact: Sky is blue.")
    assert r.text == "hi"
    assert fake.calls == 1


async def test_search_client_returns_formatted_chunks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["api-key"] == "search-secret"
        return httpx.Response(
            200,
            json={
                "value": [
                    {
                        "metadata_storage_name": "doc1.txt",
                        "content": "Tallest building is Burj Khalifa.",
                    },
                    {"id": "2", "custom_field": "Height is 828m."},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        s = make_settings(
            azure_search_endpoint="https://learnaisearch24.search.windows.net",
            azure_search_api_key=SecretStr("search-secret"),
            azure_search_index="azureblob-index",
        )
        client = SearchClient(s, http_client=http_client)
        assert client.is_configured is True
        ctx = await client.search("tallest building")
        assert "[Source: doc1.txt]\nTallest building is Burj Khalifa." in ctx
        assert "[Source: 2]\ncustom_field: Height is 828m." in ctx


async def test_search_client_unconfigured_returns_empty() -> None:
    client = SearchClient(make_settings())
    assert client.is_configured is False
    assert await client.search("anything") == ""


async def test_qdrant_ingest_and_search(tmp_path: Any) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "products.json").write_text(
        '[{"product_id": "1", "name": "Nike Running Shoes", "price": 2500}]',
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text("Gaming Laptop high performance", encoding="utf-8")

    records = load_docs_from_folder(docs_dir)
    assert len(records) == 2

    stored_points: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT" and request.url.path.endswith("/points"):
            import json as _json

            body = _json.loads(request.content.decode("utf-8"))
            stored_points.extend(body["points"])
            return httpx.Response(200, json={"status": "ok"})
        if request.method in ("PUT", "DELETE"):
            return httpx.Response(200, json={"status": "ok"})
        if request.method == "POST" and request.url.path.endswith("/points/search"):
            return httpx.Response(
                200,
                json={
                    "result": [
                        {"id": p["id"], "score": 0.85, "payload": p["payload"]}
                        for p in stored_points
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        s = make_settings(docs_dir=str(docs_dir), qdrant_collection="test_col")
        qdrant = QdrantService(s, http_client=http_client)
        ctx = await qdrant.search("Nike Running Shoes detail?")
        assert "Nike Running Shoes" in ctx
        assert "2500" in ctx
        # Unrelated query with no overlapping keywords returns empty string
        no_match = await qdrant.search("Apple MacBook warranty")
        assert no_match == ""


async def test_azure_sync_and_verify(tmp_path: Any) -> None:
    import base64

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    sample = '{"products": [{"id": 1, "title": "Mascara", "category": "beauty", "price": 9.99}]}'
    (docs_dir / "sproduct.json").write_text(sample, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        if "blob.core.windows.net" in str(request.url):
            return httpx.Response(201)
        if request.method == "GET" and "/indexes/" in request.url.path:
            return httpx.Response(404)
        if request.method == "PUT" and "/indexes/" in request.url.path:
            return httpx.Response(201)
        if request.method == "POST" and request.url.path.endswith("/docs/index"):
            return httpx.Response(200, json={"value": []})
        if request.method == "POST" and request.url.path.endswith("/docs/search"):
            return httpx.Response(200, json={"@search.count": 1, "value": [{"id": "1"}]})
        return httpx.Response(404)

    fake_key = base64.b64encode(b"0123456789abcdef0123456789abcdef").decode("utf-8")
    conn_str = (
        f"DefaultEndpointsProtocol=https;AccountName=aisearchlearn24;"
        f"AccountKey={fake_key};EndpointSuffix=core.windows.net"
    )
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        s = make_settings(
            docs_dir=str(docs_dir),
            azure_search_endpoint="https://learnaisearch24.search.windows.net",
            azure_search_api_key=SecretStr("admin-key"),
            azure_search_index="test-index",
            azure_storage_connection_string=SecretStr(conn_str),
            azure_storage_container="product",
        )
        svc = AzureSyncService(s, http_client=http_client)
        res = await svc.sync_and_verify()
        assert res["blobs_uploaded"] == ["sproduct.json"]
        assert res["documents_indexed"] == 1
        assert res["verified_search_hits"] == 1
        assert AzureSyncError is not None
