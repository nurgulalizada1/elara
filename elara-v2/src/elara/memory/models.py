from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class MemoryKind(StrEnum):
    PREFERENCE = "preference"  # affects how ELARA behaves (style, language, likes)
    FACT = "fact"              # stable facts about the user (name, job, favourite X)
    EPISODIC = "episodic"      # dated/temporary events (exam tomorrow); expires


class MemorySource(StrEnum):
    USER_EXPLICIT = "user_explicit"            # "remember that ..."
    USER_STATEMENT = "user_statement"          # self-disclosure detected in a user message
    ASSISTANT_INFERRED = "assistant_inferred"  # model asked to store it via memory_store tool
    API = "api"                                # direct API/CLI call by the user


class Memory(BaseModel):
    id: int
    kind: MemoryKind
    key: str | None
    content: str
    source: MemorySource
    confidence: float
    conversation_id: str | None
    origin_message_id: int | None
    created_at: str
    updated_at: str
    expires_at: str | None


class MemoryCandidate(BaseModel):
    kind: MemoryKind
    content: str
    key: str | None = None
    source: MemorySource = MemorySource.USER_EXPLICIT
    confidence: float = 1.0
    expires_at: str | None = None
    reason: str = ""


class SaveResult(BaseModel):
    action: str  # created | updated | duplicate
    memory: Memory
    previous: Memory | None = None
