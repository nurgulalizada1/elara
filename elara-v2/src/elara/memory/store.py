"""SQLite-backed memory store with FTS5 search.

`MemoryIndex`-style search is kept behind `search()` so a vector backend (e.g.
sqlite-vec) can later be added without changing callers.
"""

from __future__ import annotations

import re
import sqlite3

from elara.core.timeutil import iso
from elara.database.db import Database
from elara.memory.models import Memory, MemoryCandidate, MemoryKind, SaveResult

_ACTIVE = ("deleted_at IS NULL AND superseded_by IS NULL AND "
           "(expires_at IS NULL OR expires_at > ?)")
_TOKEN = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {
    # en
    "the", "a", "an", "is", "are", "was", "what", "whats", "my", "me", "i", "you", "do", "does",
    "about", "that", "this", "to", "of", "and", "or", "in", "on", "for", "remember", "know",
    "s", "it", "your", "which", "who",
    # az
    "mənim", "nə", "nədir", "idi", "bu", "o", "və", "ki", "haqqında", "haqqımda", "mən", "sən",
    "bilirsən", "yadda", "saxla", "hansı", "kim",
    # tr
    "benim", "ne", "nedir", "bir", "ve", "hakkında", "hakkımda", "ben", "sen", "biliyorsun",
    "hangi",
}


def fts_query(text: str) -> str | None:
    terms = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        # Cheap stemming for agglutinative az/tr: match on a prefix of long words.
        stem = tok[:5] if len(tok) > 6 else tok
        terms.append(f'"{stem}"*')
    return " OR ".join(dict.fromkeys(terms)) or None


def normalize_key(key: str | None) -> str | None:
    if not key:
        return None
    key = re.sub(r"\s+", " ", key.strip().lower())
    return key[:80] or None


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _row(r: sqlite3.Row) -> Memory:
        return Memory(id=r["id"], kind=r["kind"], key=r["key"], content=r["content"],
                      source=r["source"], confidence=r["confidence"],
                      conversation_id=r["conversation_id"],
                      origin_message_id=r["origin_message_id"], created_at=r["created_at"],
                      updated_at=r["updated_at"], expires_at=r["expires_at"])

    def add(self, cand: MemoryCandidate, *, conversation_id: str | None = None,
            origin_message_id: int | None = None) -> SaveResult:
        key = normalize_key(cand.key)
        now = iso()
        with self.db.transaction() as c:
            previous = None
            if key:
                r = c.execute(f"SELECT * FROM memories WHERE key=? AND {_ACTIVE}"
                              " ORDER BY id DESC LIMIT 1", (key, now)).fetchone()
                previous = self._row(r) if r else None
            else:
                r = c.execute(f"SELECT * FROM memories WHERE lower(content)=lower(?) AND {_ACTIVE}",
                              (cand.content, now)).fetchone()
                if r:
                    return SaveResult(action="duplicate", memory=self._row(r))
            if previous and previous.content.casefold() == cand.content.casefold():
                return SaveResult(action="duplicate", memory=previous)
            cur = c.execute(
                "INSERT INTO memories(kind, key, content, source, conversation_id,"
                " origin_message_id, confidence, created_at, updated_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (cand.kind.value, key, cand.content, cand.source.value, conversation_id,
                 origin_message_id, cand.confidence, now, now, cand.expires_at))
            new_id = int(cur.lastrowid)
            if previous:
                c.execute("UPDATE memories SET superseded_by=? WHERE id=?", (new_id, previous.id))
            memory = self._row(c.execute("SELECT * FROM memories WHERE id=?",
                                         (new_id,)).fetchone())
        return SaveResult(action="updated" if previous else "created", memory=memory,
                          previous=previous)

    def get(self, memory_id: int) -> Memory | None:
        with self.db.connect() as c:
            r = c.execute("SELECT * FROM memories WHERE id=? AND deleted_at IS NULL",
                          (memory_id,)).fetchone()
        return self._row(r) if r else None

    def update(self, memory_id: int, content: str) -> Memory | None:
        """Correct a memory in place (user-initiated)."""
        with self.db.connect() as c:
            cur = c.execute("UPDATE memories SET content=?, updated_at=?, confidence=1.0"
                            " WHERE id=? AND deleted_at IS NULL", (content, iso(), memory_id))
            if cur.rowcount == 0:
                return None
        return self.get(memory_id)

    def delete(self, memory_id: int) -> bool:
        """Hard delete (privacy): removes the memory and the versions it superseded."""
        with self.db.transaction() as c:
            ids = [memory_id]
            frontier = [memory_id]
            while frontier:
                rows = c.execute(
                    f"SELECT id FROM memories WHERE superseded_by IN ({','.join('?' * len(frontier))})",
                    frontier).fetchall()
                frontier = [r[0] for r in rows]
                ids.extend(frontier)
            c.execute(f"UPDATE memories SET superseded_by=NULL WHERE id IN "
                      f"({','.join('?' * len(ids))})", ids)
            cur = c.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)
            return cur.rowcount > 0

    def list(self, kind: MemoryKind | None = None, limit: int = 100) -> list[Memory]:
        q = f"SELECT * FROM memories WHERE {_ACTIVE}"
        params: list = [iso()]
        if kind:
            q += " AND kind=?"
            params.append(kind.value)
        q += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.db.connect() as c:
            return [self._row(r) for r in c.execute(q, params)]

    def search(self, query: str, limit: int = 5) -> list[Memory]:
        fq = fts_query(query)
        if not fq:
            return []
        with self.db.connect() as c:
            rows = c.execute(
                f"SELECT m.* FROM memories_fts f JOIN memories m ON m.id=f.rowid"
                f" WHERE memories_fts MATCH ? AND {_ACTIVE.replace('deleted_at', 'm.deleted_at')}"
                " ORDER BY bm25(memories_fts) LIMIT ?", (fq, iso(), limit)).fetchall()
        return [self._row(r) for r in rows]

    def by_key(self, key: str) -> Memory | None:
        with self.db.connect() as c:
            r = c.execute(f"SELECT * FROM memories WHERE key=? AND {_ACTIVE} ORDER BY id DESC",
                          (normalize_key(key), iso())).fetchone()
        return self._row(r) if r else None

    def purge_expired(self) -> int:
        with self.db.connect() as c:
            return c.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                             (iso(),)).rowcount
