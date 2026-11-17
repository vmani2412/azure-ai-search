from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=1000, ge=1, le=4096)


class ChatResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    retries: int = 0


class IngestResponse(BaseModel):
    status: str
    collection: str
    points_indexed: int


class AzureSyncResponse(BaseModel):
    status: str
    storage_container: str
    blobs_uploaded: list[str]
    search_index: str
    documents_indexed: int
    verified_search_hits: int
