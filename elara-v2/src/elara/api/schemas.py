from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, max_length=64)


class ConfirmRequest(BaseModel):
    approve: bool


class MemorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class MemoryStoreRequest(BaseModel):
    content: str = Field(min_length=3, max_length=500)
    kind: Literal["preference", "fact", "episodic"] | None = None
    key: str | None = Field(default=None, max_length=80)


class MemoryUpdateRequest(BaseModel):
    content: str = Field(min_length=3, max_length=500)


class MemoryOut(BaseModel):
    id: int
    kind: str
    key: str | None
    content: str
    source: str
    confidence: float
    created_at: str
    updated_at: str
    expires_at: str | None
    conversation_id: str | None


class MemoryStoreResponse(BaseModel):
    action: str
    memory: MemoryOut
    previous: MemoryOut | None = None


class ToolRunRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = Field(default=False, description="Set true to explicitly approve an "
                                                       "action that requires confirmation")


class ResearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    sources: list[str] | None = None
    synthesize: bool = True
    language: Literal["az", "en", "tr"] = "en"


class ErrorBody(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
