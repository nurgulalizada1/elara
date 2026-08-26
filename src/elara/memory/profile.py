"""Structured user information stored by ELARA."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserProfile:
    """Structured facts about the user."""

    name: str | None = None
    pending_name: str | None = None
