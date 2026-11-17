"""Thin wrapper around Azure OpenAI with 429 retry and content-filter handling.

The rest of the app depends on LLMClient, never on the SDK directly. That keeps
prompt/model portability (Azure OpenAI / Claude / Gemini) a one-file change.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import openai
from openai import AsyncAzureOpenAI

from app.config import Settings

log = logging.getLogger(__name__)

NO_DATA_MESSAGE = "Requested data is not available."

GROUNDED_SYSTEM_PROMPT = (
    "You are a strict data-grounded assistant. "
    "Answer the user's question ONLY using the facts in the PROVIDED CONTEXT below. "
    "Do NOT use outside knowledge, assumptions, or general training data. "
    "If the answer is not explicitly present in the PROVIDED CONTEXT, "
    f"reply with the exact phrase: '{NO_DATA_MESSAGE}'"
)


class RateLimitExhaustedError(Exception):
    """429 persisted after all retries."""

    def __init__(self, retry_after_s: float | None) -> None:
        super().__init__("Rate limit persisted after retries")
        self.retry_after_s = retry_after_s


class ContentFilterError(Exception):
    """Azure content filter blocked the prompt or the completion."""

    def __init__(self, categories: dict[str, Any]) -> None:
        super().__init__("Blocked by content filter")
        # category -> {filtered, severity}. Never the offending content itself.
        self.categories = categories


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    retries: int


class _Completions(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


def _retry_after_seconds(exc: openai.RateLimitError) -> float | None:
    """Prefer retry-after-ms, then retry-after (seconds)."""
    headers = exc.response.headers
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return float(ms) / 1000
        except ValueError:
            pass
    s = headers.get("retry-after")
    if s:
        try:
            return float(s)
        except ValueError:
            return None
    return None


def backoff_delay(attempt: int, base: float, cap: float, rng: random.Random) -> float:
    """Exponential backoff with full jitter: uniform(0, min(cap, base * 2**attempt))."""
    return rng.uniform(0, min(cap, base * (2**attempt)))


def _filter_categories(exc: openai.BadRequestError) -> dict[str, Any] | None:
    body = exc.body if isinstance(exc.body, dict) else {}
    if body.get("code") != "content_filter":
        return None
    result = (body.get("innererror") or {}).get("content_filter_result") or {}
    return {k: v for k, v in result.items() if isinstance(v, dict)}


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        completions: _Completions | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._s = settings
        resolved: Any = completions
        if resolved is None:
            sdk = AsyncAzureOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key.get_secret_value(),
                api_version=settings.azure_openai_api_version,
                max_retries=0,  # we own the retry policy so we can measure it
                timeout=settings.llm_timeout_s,
            )
            resolved = sdk.chat.completions
        self._completions: _Completions = resolved
        self._sleep = sleep
        self._rng = rng or random.Random()

    async def chat(
        self,
        message: str,
        max_tokens: int = 1000,
        context: str | None = None,
    ) -> LLMResult:
        if context is not None and not context.strip():
            log.info("grounded_search_no_data")
            return LLMResult(
                text=NO_DATA_MESSAGE,
                prompt_tokens=0,
                completion_tokens=0,
                retries=0,
            )

        if context is not None:
            messages: list[dict[str, str]] = [
                {
                    "role": "system",
                    "content": f"{GROUNDED_SYSTEM_PROMPT}\n\nPROVIDED CONTEXT:\n{context}",
                },
                {"role": "user", "content": message},
            ]
        else:
            messages = [{"role": "user", "content": message}]

        retries = 0
        last_retry_after: float | None = None
        for attempt in range(self._s.llm_max_retries + 1):
            try:
                resp = await self._completions.create(
                    model=self._s.azure_openai_deployment,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
            except openai.RateLimitError as exc:
                last_retry_after = _retry_after_seconds(exc)
                if attempt == self._s.llm_max_retries:
                    break
                delay = (
                    last_retry_after
                    if last_retry_after is not None
                    else backoff_delay(
                        attempt, self._s.llm_base_delay_s, self._s.llm_max_delay_s, self._rng
                    )
                )
                retries += 1
                log.warning(
                    "azure_openai_429",
                    extra={
                        "ctx": {
                            "attempt": attempt + 1,
                            "sleep_s": round(delay, 2),
                            "used_retry_after": last_retry_after is not None,
                        }
                    },
                )
                await self._sleep(delay)
                continue
            except openai.BadRequestError as exc:
                categories = _filter_categories(exc)
                if categories is not None:
                    # Log categories and severities only, never the offending text.
                    log.warning("content_filter_blocked", extra={"ctx": {"categories": categories}})
                    raise ContentFilterError(categories) from exc
                raise

            choice = resp.choices[0]
            if choice.finish_reason == "content_filter":
                log.warning("content_filter_on_output")
                raise ContentFilterError({"output": {"filtered": True}})
            return LLMResult(
                text=choice.message.content or "",
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                retries=retries,
            )
        raise RateLimitExhaustedError(last_retry_after)
