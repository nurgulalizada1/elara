"""HTTP layer for research sources: retries with backoff, per-host throttling, SQLite cache."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import random
import time
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

import httpx

from elara.core.errors import SourceError
from elara.core.logging import get_logger
from elara.core.timeutil import iso, utcnow
from elara.database.db import Database

log = get_logger(__name__)
_cache_flags: contextvars.ContextVar[list[bool] | None] = contextvars.ContextVar(
    "research_cache_flags", default=None)
RETRY_STATUS = {429, 500, 502, 503, 504}


class ResearchHttp:
    def __init__(self, client: httpx.AsyncClient, db: Database | None = None, *,
                 cache_ttl_s: int = 6 * 3600, timeout_s: float = 20.0, max_retries: int = 2,
                 backoff_s: float = 0.6, min_intervals: dict[str, float] | None = None,
                 contact_email: str | None = None):
        self.client = client
        self.db = db
        self.cache_ttl_s = cache_ttl_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.min_intervals = min_intervals or {}
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._cooldown_until: dict[str, float] = {}
        ua = "ELARA/0.1 (personal research assistant)"
        self.user_agent = f"{ua}; mailto:{contact_email}" if contact_email else ua

    async def get_json(self, url: str, *, params: dict | None = None,
                       headers: dict | None = None, source: str = "http") -> Any:
        return await self._request("GET", url, params=params, headers=headers, source=source)

    async def post_json(self, url: str, *, body: dict, headers: dict | None = None,
                        source: str = "http") -> Any:
        return await self._request("POST", url, body=body, headers=headers, source=source)

    def _cache_key(self, method: str, url: str, params: dict | None, body: dict | None) -> str:
        raw = json.dumps([method, url, sorted((params or {}).items()), body], sort_keys=True,
                         default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Any | None:
        if not self.db:
            return None
        with self.db.connect() as c:
            r = c.execute("SELECT body FROM http_cache WHERE key=? AND expires_at > ?",
                          (key, iso())).fetchone()
        return json.loads(r[0]) if r else None

    def _cache_put(self, key: str, data: Any) -> None:
        if not self.db or self.cache_ttl_s <= 0:
            return
        with self.db.connect() as c:
            c.execute("INSERT OR REPLACE INTO http_cache(key, status, body, created_at, expires_at)"
                      " VALUES (?,?,?,?,?)", (key, 200, json.dumps(data), iso(),
                                              iso(utcnow() + timedelta(seconds=self.cache_ttl_s))))

    async def _throttle(self, host: str) -> None:
        interval = self.min_intervals.get(host)
        if not interval:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._last.get(host, 0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[host] = time.monotonic()

    def _cooling_down(self, host: str) -> float:
        return max(0.0, self._cooldown_until.get(host, 0.0) - time.monotonic())

    async def _request(self, method: str, url: str, *, params=None, body=None, headers=None,
                       source: str) -> Any:
        key = self._cache_key(method, url, params, body)
        cached = self._cache_get(key)
        flags = _cache_flags.get()
        if cached is not None:
            if flags is not None:
                flags.append(True)
            return cached
        if flags is not None:
            flags.append(False)
        host = urlsplit(url).hostname or ""
        if (wait := self._cooling_down(host)) > 0:
            # Respect a previous 429: don't hit the API again until its window has passed.
            raise SourceError(f"{source}: rate limited; not retrying for another {wait:.0f}s",
                              "rate_limited")
        hdrs = {"user-agent": self.user_agent, "accept": "application/json", **(headers or {})}
        attempt = 0
        while True:
            await self._throttle(host)
            try:
                resp = await self.client.request(method, url, params=params, json=body,
                                                 headers=hdrs, timeout=self.timeout_s)
            except httpx.TimeoutException as e:
                err = SourceError(f"{source}: timed out", "timeout")
                retry, delay = True, None
                cause: Exception = e
            except httpx.TransportError as e:
                err = SourceError(f"{source}: unreachable ({type(e).__name__})", "unreachable")
                retry, delay = True, None
                cause = e
            else:
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        raise SourceError(f"{source}: invalid JSON", "invalid_response") from e
                    self._cache_put(key, data)
                    return data
                status = resp.status_code
                ra = resp.headers.get("retry-after", "")
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else None
                if status == 429:
                    err = SourceError(f"{source}: HTTP 429 (rate limited)", "rate_limited")
                    # Long server-requested waits are not slept through: fail now, cool down.
                    retry = delay is None or delay <= 5
                elif status in (401, 403):
                    err = SourceError(f"{source}: HTTP {status} (access blocked or key rejected)",
                                      "blocked")
                    retry = False
                else:
                    err = SourceError(f"{source}: HTTP {status}", "http_error")
                    retry = status in RETRY_STATUS
                cause = err
            if not retry or attempt >= self.max_retries:
                if err.kind == "rate_limited":
                    self._cooldown_until[host] = time.monotonic() + min(max(delay or 60, 5), 3600)
                log.warning("research.source_failed", extra={"source": source, "error": str(err),
                                                             "kind": err.kind})
                raise err from cause
            sleep = min(delay or self.backoff_s * (2 ** attempt) + random.uniform(0, 0.2), 5.0)
            await asyncio.sleep(sleep)
            attempt += 1


def track_cache() -> contextvars.Token:
    """Start recording (per asyncio task) whether responses came from the cache."""
    return _cache_flags.set([])


def cache_flags() -> list[bool]:
    return list(_cache_flags.get() or [])
