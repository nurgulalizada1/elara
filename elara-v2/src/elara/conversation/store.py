"""Persistence for conversations, messages and per-conversation working state."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from elara.core.context import new_id
from elara.core.timeutil import iso
from elara.database.db import Database


class StoredMessage(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    language: str | None = None
    intent: str | None = None
    tier: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ConversationStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, title: str | None = None) -> str:
        cid = new_id("conv_")
        now = iso()
        with self.db.connect() as c:
            c.execute("INSERT INTO conversations(id, title, created_at, updated_at)"
                      " VALUES (?,?,?,?)", (cid, title, now, now))
        return cid

    def exists(self, conversation_id: str) -> bool:
        with self.db.connect() as c:
            return c.execute("SELECT 1 FROM conversations WHERE id=?",
                             (conversation_id,)).fetchone() is not None

    def ensure(self, conversation_id: str | None) -> str:
        if conversation_id and self.exists(conversation_id):
            return conversation_id
        return self.create()

    def add_message(self, conversation_id: str, role: str, content: str, *,
                    language: str | None = None, intent: str | None = None,
                    tier: int | None = None, metadata: dict | None = None) -> int:
        now = iso()
        with self.db.transaction() as c:
            cur = c.execute(
                "INSERT INTO messages(conversation_id, role, content, language, intent, tier,"
                " metadata_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (conversation_id, role, content, language, intent, tier,
                 json.dumps(metadata or {}, ensure_ascii=False, default=str), now))
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
            return int(cur.lastrowid)

    def recent_messages(self, conversation_id: str, limit: int = 20) -> list[StoredMessage]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY id DESC"
                " LIMIT ?) ORDER BY id", (conversation_id, limit)).fetchall()
        return [StoredMessage(id=r["id"], conversation_id=r["conversation_id"], role=r["role"],
                              content=r["content"], language=r["language"], intent=r["intent"],
                              tier=r["tier"], metadata=json.loads(r["metadata_json"]),
                              created_at=r["created_at"]) for r in rows]

    def get_state(self, conversation_id: str) -> dict[str, Any]:
        with self.db.connect() as c:
            row = c.execute("SELECT state_json FROM conversations WHERE id=?",
                            (conversation_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def set_state(self, conversation_id: str, state: dict[str, Any]) -> None:
        with self.db.connect() as c:
            c.execute("UPDATE conversations SET state_json=?, updated_at=? WHERE id=?",
                      (json.dumps(state, ensure_ascii=False, default=str), iso(),
                       conversation_id))

    def clear(self, conversation_id: str) -> None:
        """Forget the conversation's history and working state (long-term memory untouched)."""
        with self.db.transaction() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            c.execute("UPDATE conversations SET state_json='{}' WHERE id=?", (conversation_id,))

    def list(self, limit: int = 20) -> list[dict]:
        with self.db.connect() as c:
            rows = c.execute(
                "SELECT c.id, c.title, c.updated_at, count(m.id) AS n FROM conversations c"
                " LEFT JOIN messages m ON m.conversation_id=c.id GROUP BY c.id"
                " ORDER BY c.updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
