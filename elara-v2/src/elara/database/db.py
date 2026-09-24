"""SQLite access: connection factory + versioned, non-destructive migrations.

A new connection is opened per unit of work (cheap for SQLite, and safe across the
threads FastAPI/asyncio may use). WAL mode allows concurrent readers.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from elara.core.errors import DatabaseError
from elara.core.logging import get_logger
from elara.core.timeutil import iso

log = get_logger(__name__)


def _load_migrations() -> list[tuple[int, str, str]]:
    out = []
    pkg = resources.files("elara.database") / "migrations"
    for entry in pkg.iterdir():
        name = entry.name
        if name.endswith(".sql") and name[:4].isdigit():
            out.append((int(name[:4]), name, entry.read_text(encoding="utf-8")))
    return sorted(out)


class Database:
    def __init__(self, path: Path):
        self.path = path

    def _open(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        except (sqlite3.Error, OSError) as e:
            raise DatabaseError(f"cannot open database {self.path}: {e}") from e
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Autocommit connection for simple reads/single writes."""
        conn = self._open()
        try:
            yield conn
        except sqlite3.Error as e:
            log.error("db.error", extra={"error": str(e)})
            raise DatabaseError(str(e)) from e
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._open()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception as e:
            conn.execute("ROLLBACK")
            if isinstance(e, sqlite3.Error):
                log.error("db.error", extra={"error": str(e)})
                raise DatabaseError(str(e)) from e
            raise
        finally:
            conn.close()

    def migrate(self) -> list[int]:
        """Apply pending migrations in order. Never drops existing data."""
        conn = self._open()
        applied: list[int] = []
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
            for version, name, sql in _load_migrations():
                if version in done:
                    continue
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    for stmt in _split_sql(sql):
                        conn.execute(stmt)
                    conn.execute(
                        "INSERT INTO schema_migrations(version, name, applied_at) VALUES (?,?,?)",
                        (version, name, iso()),
                    )
                    conn.execute("COMMIT")
                except sqlite3.Error as e:
                    conn.execute("ROLLBACK")
                    raise DatabaseError(f"migration {name} failed: {e}") from e
                applied.append(version)
                log.info("db.migrated", extra={"version": version, "migration": name})
        except sqlite3.Error as e:
            raise DatabaseError(str(e)) from e
        finally:
            conn.close()
        return applied

    def schema_version(self) -> int:
        with self.connect() as c:
            row = c.execute("SELECT max(version) FROM schema_migrations").fetchone()
            return int(row[0] or 0)

    def check(self) -> None:
        """Raise DatabaseError if the database is not usable."""
        with self.connect() as c:
            res = c.execute("PRAGMA quick_check").fetchone()[0]
            if res != "ok":
                raise DatabaseError(f"integrity check failed: {res}")


def _split_sql(sql: str) -> list[str]:
    """Split a migration script into statements, keeping trigger bodies intact."""
    stmts, buf = [], []
    for line in sql.splitlines():
        stripped = line.strip()
        if not buf and (not stripped or stripped.startswith("--")):
            continue
        buf.append(line)
        candidate = "\n".join(buf)
        if stripped.endswith(";") and sqlite3.complete_statement(candidate):
            stmts.append(candidate)
            buf = []
    if buf and "\n".join(buf).strip():
        stmts.append("\n".join(buf))
    return stmts
