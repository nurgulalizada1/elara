"""Pending actions awaiting explicit user confirmation (persisted, expiring)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from elara.core.context import new_id
from elara.core.timeutil import iso, parse_iso, utcnow
from elara.database.db import Database


class PendingAction(BaseModel):
    id: str
    conversation_id: str | None
    tool_name: str
    arguments: dict[str, Any]
    reason: str
    description: str = ""
    status: str = "pending"
    expires_at: str


class ConfirmationStore:
    def __init__(self, db: Database, ttl_s: int = 600):
        self.db = db
        self.ttl_s = ttl_s

    def create(self, conversation_id: str | None, tool_name: str, arguments: dict,
               reason: str, description: str) -> PendingAction:
        pa = PendingAction(id=new_id("act_"), conversation_id=conversation_id,
                           tool_name=tool_name, arguments=arguments, reason=reason,
                           description=description,
                           expires_at=iso(utcnow() + timedelta(seconds=self.ttl_s)))
        with self.db.transaction() as c:
            # Only one pending action per conversation: a new one supersedes the old.
            if conversation_id:
                c.execute("UPDATE pending_actions SET status='cancelled' WHERE conversation_id=?"
                          " AND status='pending'", (conversation_id,))
            c.execute("INSERT INTO pending_actions(id, conversation_id, tool_name, arguments_json,"
                      " reason, status, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?)",
                      (pa.id, conversation_id, tool_name,
                       json.dumps({"args": arguments, "description": description},
                                  ensure_ascii=False), reason, "pending", iso(), pa.expires_at))
        return pa

    def _row(self, r) -> PendingAction:
        payload = json.loads(r["arguments_json"])
        return PendingAction(id=r["id"], conversation_id=r["conversation_id"],
                             tool_name=r["tool_name"], arguments=payload["args"],
                             description=payload.get("description", ""), reason=r["reason"],
                             status=r["status"], expires_at=r["expires_at"])

    def get(self, action_id: str) -> PendingAction | None:
        with self.db.connect() as c:
            r = c.execute("SELECT * FROM pending_actions WHERE id=?", (action_id,)).fetchone()
        return self._row(r) if r else None

    def pending_for(self, conversation_id: str) -> PendingAction | None:
        with self.db.connect() as c:
            r = c.execute("SELECT * FROM pending_actions WHERE conversation_id=? AND"
                          " status='pending' ORDER BY created_at DESC LIMIT 1",
                          (conversation_id,)).fetchone()
        return self._row(r) if r else None

    def is_expired(self, pa: PendingAction) -> bool:
        exp = parse_iso(pa.expires_at)
        return exp is not None and exp < utcnow()

    def resolve(self, action_id: str, status: str) -> bool:
        """Atomically move a pending action to a final state. False if already resolved."""
        with self.db.connect() as c:
            cur = c.execute("UPDATE pending_actions SET status=? WHERE id=? AND status='pending'",
                            (status, action_id))
            return cur.rowcount == 1
