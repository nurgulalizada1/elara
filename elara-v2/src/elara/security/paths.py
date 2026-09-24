"""Filesystem jail: every file operation goes through PathGuard.resolve()."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Literal

from elara.core.errors import PathNotAllowed

SENSITIVE_DIRS = [
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".password-store",
    ".config/gcloud", ".local/share/keyrings", ".mozilla", ".config/google-chrome",
    ".config/chromium", ".config/BraveSoftware", ".thunderbird", ".pki", ".netrc",
    ".git-credentials", ".config/gh",
]
SENSITIVE_NAMES = ["*.pem", "*.key", "id_rsa*", "id_ed25519*", "id_ecdsa*", ".env", ".env.*",
                   "*.kdbx", "credentials*", "*.p12", "*.pfx", ".pgpass", "secrets.*"]


class PathGuard:
    def __init__(self, read_dirs: list[Path], write_dirs: list[Path], *,
                 deny_dirs: list[Path] | None = None, home: Path | None = None):
        self.read_dirs = [p.resolve() for p in read_dirs]
        self.write_dirs = [p.resolve() for p in write_dirs]
        home = (home or Path.home()).resolve()
        self.deny_dirs = [(home / d).resolve() for d in SENSITIVE_DIRS] + [
            p.resolve() for p in (deny_dirs or [])]

    @property
    def default_dir(self) -> Path:
        return (self.write_dirs or self.read_dirs)[0]

    def resolve(self, raw: str | Path, mode: Literal["read", "write"] = "read") -> Path:
        s = str(raw)
        if "\x00" in s:
            raise PathNotAllowed("path contains a NUL byte")
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = self.default_dir / p
        resolved = p.resolve()  # collapses '..' and follows symlinks
        roots = self.write_dirs if mode == "write" else self.read_dirs + self.write_dirs
        if not any(resolved == r or resolved.is_relative_to(r) for r in roots):
            allowed = ", ".join(str(r) for r in roots)
            raise PathNotAllowed(f"{resolved} is outside the allowed {mode} directories ({allowed})")
        for d in self.deny_dirs:
            if resolved == d or resolved.is_relative_to(d):
                raise PathNotAllowed(f"{resolved} is in a protected location")
        if any(fnmatch.fnmatch(resolved.name, pat) for pat in SENSITIVE_NAMES):
            raise PathNotAllowed(f"{resolved.name} looks like a credential file")
        return resolved
