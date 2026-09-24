"""HTTP layer for research sources: retries with backoff, per-host throttling, SQLite cache."""

from __future__ import annotations

import asyncio
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

    async def _request(self, method: str, url: str, *, params=None, body=None, headers=None,
                       source: str) -> Any:
        key = self._cache_key(method, url, params, body)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        host = urlsplit(url).hostname or ""
        hdrs = {"user-agent": self.user_agent, "accept": "application/json", **(headers or {})}
        attempt = 0
        while True:
            await self._throttle(host)
            try:
                resp = await self.client.request(method, url, params=params, json=body,
                                                 headers=hdrs, timeout=self.timeout_s)
            except httpx.TimeoutException as e:
                err = SourceError(f"{source}: timed out")
                retry, delay = True, None
                cause: Exception = e
            except httpx.TransportError as e:
                err = SourceError(f"{source}: unreachable ({type(e).__name__})")
                retry, delay = True, None
                cause = e
            else:
                if resp.status_code < 400:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        raise SourceError(f"{source}: invalid JSON") from e
                    self._cache_put(key, data)
                    return data
                err = SourceError(f"{source}: HTTP {resp.status_code}")
                retry = resp.status_code in RETRY_STATUS
                ra = resp.headers.get("retry-after", "")
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else None
                cause = err
            if not retry or attempt >= self.max_retries:
                log.warning("research.source_failed", extra={"source": source, "error": str(err)})
                raise err from cause
            sleep = min(delay or self.backoff_s * (2 ** attempt) + random.uniform(0, 0.2), 5.0)
            await asyncio.sleep(sleep)
            attempt += 1
