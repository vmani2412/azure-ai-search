from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=1000, ge=1, le=4096)


class ChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    retries: int = 0
