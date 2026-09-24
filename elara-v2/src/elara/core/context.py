"""Request-scoped context (request id, conversation id) carried via contextvars."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


@contextmanager
def request_scope(request_id: str | None = None, conversation_id: str | None = None):
    rt = request_id_var.set(request_id or new_id("req_"))
    ct = conversation_id_var.set(conversation_id)
    try:
        yield request_id_var.get()
    finally:
        request_id_var.reset(rt)
        conversation_id_var.reset(ct)
