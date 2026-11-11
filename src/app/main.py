import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.logging_conf import request_id_var, setup_logging
from app.schemas import ChatRequest, ChatResponse
from app.services.llm import ContentFilterError, LLMClient, RateLimitExhaustedError

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.llm = LLMClient(settings)  # one client per process, reused across requests
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


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, llm: Annotated[LLMClient, Depends(get_llm)]) -> ChatResponse:
    result = await llm.chat(req.message, req.max_tokens)
    return ChatResponse(
        answer=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retries=result.retries,
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
