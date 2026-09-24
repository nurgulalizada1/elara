"""Validation of text the user sends to ELARA."""

from __future__ import annotations

import re

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_user_text(text: str) -> str:
    """Strip control characters (keeps newlines/tabs) and surrounding whitespace."""
    return _CONTROL.sub("", text).strip()
