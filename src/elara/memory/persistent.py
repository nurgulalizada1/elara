"""Persistent user memory for ELARA."""

from __future__ import annotations

import json
from pathlib import Path

from elara.memory.profile import UserProfile


class PersistentMemory:
    """Save and load ELARA's structured user profile."""

    def __init__(self, path: str | Path = ".elara_memory.json") -> None:
        self.path = Path(path)

    def load_profile(self) -> UserProfile:
        """Load a profile from disk, or return an empty profile."""
        if not self.path.exists():
            return UserProfile()

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return UserProfile()

        if not isinstance(data, dict):
            return UserProfile()

        return UserProfile(
            name=data.get("name"),
            pending_name=data.get("pending_name"),
        )

    def save_profile(self, profile: UserProfile) -> None:
        """Save a profile to disk."""
        data = {
            "name": profile.name,
            "pending_name": profile.pending_name,
        }

        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
