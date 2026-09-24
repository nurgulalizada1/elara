"""Structured JSON logging with secret redaction.

Usage: ``log = get_logger(__name__); log.info("tool.call", extra={"tool": "x"})``.
Every record carries request_id / conversation_id from contextvars. Values of all
configured secrets plus common credential patterns are redacted before output.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from elara.core.context import conversation_id_var, request_id_var

_STD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|password|passwd|secret|token)\s*[=:]\s*)[^\s,;\"']+"),
]


class Redactor:
    def __init__(self, secrets: list[str] | None = None):
        self._secrets = [s for s in (secrets or []) if len(s) >= 6]

    def add(self, secret: str) -> None:
        if len(secret) >= 6:
            self._secrets.append(secret)

    def __call__(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "[REDACTED]")
        for pat in _SECRET_PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
            else:
                text = pat.sub("[REDACTED]", text)
        return text


_redactor = Redactor()


def redact(text: str) -> str:
    return _redactor(text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
            "conversation_id": conversation_id_var.get(),
        }
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _redactor(json.dumps(payload, default=str, ensure_ascii=False))


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items()
                  if k not in _STD_ATTRS and not k.startswith("_")}
        base = f"{record.levelname.lower():7} {record.name}: {record.getMessage()}"
        if extras:
            base += " " + json.dumps(extras, default=str, ensure_ascii=False)
        return _redactor(base)


def configure_logging(
    level: str = "INFO",
    *,
    json_format: bool = True,
    log_file: Path | None = None,
    console: bool = True,
    console_level: str | None = None,
    secrets: list[str] | None = None,
) -> None:
    global _redactor
    _redactor = Redactor(secrets)
    root = logging.getLogger("elara")
    root.handlers.clear()
    root.setLevel(level.upper())
    root.propagate = False
    fmt: logging.Formatter = JsonFormatter() if json_format else PlainFormatter()
    if console:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        h.setLevel((console_level or level).upper())
        root.addHandler(h)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("elara"):
        name = f"elara.{name}"
    return logging.getLogger(name)
