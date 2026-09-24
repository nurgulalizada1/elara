"""Append-only audit tables: model calls, tool calls, security events."""

from __future__ import annotations

import json
from typing import Any

from elara.core.context import conversation_id_var, request_id_var
from elara.core.logging import get_logger
from elara.core.timeutil import iso
from elara.database.db import Database

log = get_logger(__name__)


class AuditLog:
    def __init__(self, db: Database):
        self.db = db

    def model_call(self, *, provider: str, model: str, tier: str | None, purpose: str | None,
                   input_tokens: int | None, output_tokens: int | None, latency_ms: int,
                   status: str, error: str | None = None) -> None:
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO model_calls(request_id, conversation_id, provider, model, tier, purpose,"
                " input_tokens, output_tokens, latency_ms, status, error, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id_var.get(), conversation_id_var.get(), provider, model, tier, purpose,
                 input_tokens, output_tokens, latency_ms, status, error, iso()),
            )

    def tool_call(self, *, tool_name: str, arguments: dict[str, Any], origin: str, permission: str,
                  status: str, confirmed: bool, result_summary: str | None, error: str | None,
                  duration_ms: int | None) -> None:
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO tool_calls(request_id, conversation_id, tool_name, arguments_json, origin,"
                " permission, status, confirmed, result_summary, error, duration_ms, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id_var.get(), conversation_id_var.get(), tool_name,
                 json.dumps(arguments, ensure_ascii=False, default=str)[:4000], origin, permission,
                 status, int(confirmed), (result_summary or "")[:500] or None, error, duration_ms,
                 iso()),
            )

    def security_event(self, kind: str, severity: str, detail: str) -> None:
        log.warning("security.event", extra={"kind": kind, "severity": severity,
                                             "detail": detail[:300]})
        with self.db.connect() as c:
            c.execute(
                "INSERT INTO security_events(request_id, conversation_id, kind, severity, detail,"
                " created_at) VALUES (?,?,?,?,?,?)",
                (request_id_var.get(), conversation_id_var.get(), kind, severity, detail[:2000],
                 iso()),
            )

    def usage_totals(self) -> dict[str, int]:
        with self.db.connect() as c:
            r = c.execute("SELECT count(*), coalesce(sum(input_tokens),0),"
                          " coalesce(sum(output_tokens),0) FROM model_calls").fetchone()
        return {"calls": r[0], "input_tokens": r[1], "output_tokens": r[2]}
