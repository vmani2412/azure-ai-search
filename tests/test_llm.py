import random
from typing import Any

import httpx
import openai
import pytest
from pydantic import SecretStr

from app.config import Settings
from app.services.llm import (
    ContentFilterError,
    LLMClient,
    RateLimitExhaustedError,
    backoff_delay,
)


def make_settings(**kw: Any) -> Settings:
    base: dict[str, Any] = {
        "azure_openai_endpoint": "https://x.openai.azure.com",
        "azure_openai_api_key": SecretStr("k"),
        "azure_openai_deployment": "d",
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
