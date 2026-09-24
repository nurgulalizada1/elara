"""FastAPI backend. Same core services as the CLI (via the Container)."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import elara
from elara.agent.assistant import AssistantReply
from elara.api.schemas import (
    ChatRequest,
    ConfirmRequest,
    ErrorResponse,
    MemoryOut,
    MemorySearchRequest,
    MemoryStoreRequest,
    MemoryStoreResponse,
    MemoryUpdateRequest,
    ResearchRequest,
    ToolRunRequest,
)
from elara.config.settings import Settings, get_settings
from elara.core.container import Container, build_container
from elara.core.context import new_id, request_scope
from elara.core.errors import DatabaseError, ElaraError, PermissionDenied, ProviderError
from elara.core.logging import get_logger
from elara.memory import MemoryKind, MemorySource
from elara.security.untrusted import Trust
from elara.tools.base import Origin, ToolContext
from elara.tools.executor import Status

log = get_logger(__name__)


class BodyLimitMiddleware:
    """Pure-ASGI request body cap (works for chunked bodies too)."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return await _too_large(send)
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _too_large(send)


class _BodyTooLarge(Exception):
    pass


async def _too_large(send):
    body = b'{"error":{"type":"payload_too_large","message":"request body too large"}}'
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": body})


def _mem(m) -> MemoryOut:
    return MemoryOut(**m.model_dump())


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    state: dict[str, Container] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["c"] = container or build_container(settings)
        yield
        if container is None:
            await state["c"].aclose()

    app = FastAPI(title="ELARA", version=elara.__version__, lifespan=lifespan,
                  responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse},
                             500: {"model": ErrorResponse}})
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    def c() -> Container:
        return state["c"]

    def auth(request: Request) -> None:
        token = settings.api_token.get_secret_value() if settings.api_token else None
        if not token:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(401, "missing or invalid bearer token")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or new_id("req_")
        rid = "".join(ch for ch in rid if ch.isalnum() or ch in "_-")[:64] or new_id("req_")
        with request_scope(rid):
            response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    def _error(status: int, kind: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"type": kind,
                                                                   "message": message}})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(PermissionDenied)
    async def denied(_: Request, exc: PermissionDenied):
        return _error(403, "permission_denied", str(exc))

    @app.exception_handler(ProviderError)
    async def provider_error(_: Request, exc: ProviderError):
        return _error(503, "provider_unavailable", str(exc))

    @app.exception_handler(DatabaseError)
    async def db_error(_: Request, exc: DatabaseError):
        log.error("api.db_error", extra={"error": str(exc)})
        return _error(500, "database_error", "local database error")

    @app.exception_handler(ElaraError)
    async def elara_error(_: Request, exc: ElaraError):
        return _error(400, type(exc).__name__, str(exc))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception):
        log.exception("api.unhandled")
        return _error(500, "internal_error", "unexpected server error")

    # --------------------------------------------------------------------- routes --
    @app.get("/health")
    async def health():
        cont = c()
        try:
            cont.db.check()
            db_ok = True
        except DatabaseError:
            db_ok = False
        return {"status": "ok" if db_ok else "degraded", "version": elara.__version__,
                "database": db_ok, "schema_version": cont.db.schema_version() if db_ok else None,
                "llm_provider": cont.settings.resolved_provider(),
                "llm_available": cont.llm.available, "tools": len(cont.registry)}

    deps = [Depends(auth)]

    @app.post("/chat", response_model=AssistantReply, dependencies=deps)
    async def chat(body: ChatRequest):
        return await c().assistant.handle(body.message, body.conversation_id)

    @app.post("/confirmations/{action_id}", response_model=AssistantReply, dependencies=deps)
    async def confirm(action_id: str, body: ConfirmRequest):
        return await c().assistant.confirm(action_id, body.approve)

    @app.get("/conversations", dependencies=deps)
    async def conversations(limit: int = 20):
        return c().conversations.list(min(max(limit, 1), 100))

    @app.get("/conversations/{cid}/messages", dependencies=deps)
    async def messages(cid: str, limit: int = 50):
        if not c().conversations.exists(cid):
            raise HTTPException(404, "conversation not found")
        return [m.model_dump() for m in c().conversations.recent_messages(cid, min(limit, 500))]

    @app.get("/memory", response_model=list[MemoryOut], dependencies=deps)
    async def memory_list(kind: MemoryKind | None = None, limit: int = 100):
        return [_mem(m) for m in c().memory.store.list(kind, min(max(limit, 1), 500))]

    @app.post("/memory/search", response_model=list[MemoryOut], dependencies=deps)
    async def memory_search(body: MemorySearchRequest):
        return [_mem(m) for m in c().memory.relevant(body.query, body.limit)]

    @app.post("/memory/store", response_model=MemoryStoreResponse, dependencies=deps)
    async def memory_store(body: MemoryStoreRequest):
        out = c().memory.remember(body.content, source=MemorySource.API, trust=Trust.USER,
                                  kind=MemoryKind(body.kind) if body.kind else None,
                                  key=body.key)
        if not out.saved or out.result is None:
            raise HTTPException(422, f"not stored: {out.reason}")
        r = out.result
        return MemoryStoreResponse(action=r.action, memory=_mem(r.memory),
                                   previous=_mem(r.previous) if r.previous else None)

    @app.patch("/memory/{memory_id}", response_model=MemoryOut, dependencies=deps)
    async def memory_update(memory_id: int, body: MemoryUpdateRequest):
        report = c().memory.policy.detector.scan(body.content)
        if report.suspicious:
            raise HTTPException(422, f"not stored: instruction-like text ({report.summary()})")
        m = c().memory.store.update(memory_id, body.content)
        if m is None:
            raise HTTPException(404, "memory not found")
        return _mem(m)

    @app.delete("/memory/{memory_id}", dependencies=deps)
    async def memory_delete(memory_id: int):
        if not c().memory.store.delete(memory_id):
            raise HTTPException(404, "memory not found")
        return {"deleted": memory_id}

    @app.get("/tools", dependencies=deps)
    async def tools():
        return [t.info() for t in c().registry.all()]

    @app.post("/tools/{tool_name}", dependencies=deps)
    async def run_tool(tool_name: str, body: ToolRunRequest):
        cont = c()
        if tool_name not in cont.registry:
            raise HTTPException(404, f"unknown tool '{tool_name}'")
        outcome = await cont.executor.execute(tool_name, body.arguments, ToolContext(
            origin=Origin.API, confirmed=body.confirmed))
        payload = {"tool": tool_name, "status": outcome.status.value, "output": outcome.output,
                   "error": outcome.error, "trust": outcome.trust.value,
                   "injection_warnings": outcome.injection_categories,
                   "duration_ms": outcome.duration_ms}
        code = {Status.OK: 200, Status.INVALID: 422, Status.DENIED: 403,
                Status.NEEDS_CONFIRMATION: 409, Status.ERROR: 502}[outcome.status]
        if outcome.pending:
            payload["confirmation"] = {"reason": outcome.pending.reason,
                                       "action": outcome.pending.description,
                                       "how": "repeat the request with \"confirmed\": true"}
            cont.confirmations.resolve(outcome.pending.id, "cancelled")
        return JSONResponse(status_code=code, content=payload)

    @app.post("/research", dependencies=deps)
    async def research(body: ResearchRequest):
        cont = c()
        result = await cont.research.research(body.query, language=body.language,
                                              sources=body.sources)
        out = {"plan": result.plan.model_dump(), "records": [r.model_dump() for r in
                                                              result.records],
               "sources": [o.model_dump() for o in result.outcomes]}
        if body.synthesize:
            syn = await cont.assistant.synthesizer.synthesize(body.query, result, body.language)
            out["synthesis"] = syn.model_dump()
        return out

    @app.get("/config", dependencies=deps)
    async def config():
        return c().settings.public_view()

    @app.get("/status", dependencies=deps)
    async def status():
        cont = c()
        with cont.db.connect() as conn:
            counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: S608
                      for t in ("conversations", "messages", "memories", "tool_calls",
                                "research_queries", "security_events")}
        return {"usage": cont.audit.usage_totals(), "counts": counts}

    return app
