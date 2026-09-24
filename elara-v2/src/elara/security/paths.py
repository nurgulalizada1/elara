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


_NAME_PREFIXES = ("my ", "the ", "mənim ", "benim ")
_NAME_SUFFIXES = (" folder", " directory", " dir", " qovluğu", " qovluğum", " qovluq",
                  " klasörü", " klasörüm", " klasör")
WORKSPACE_ALIASES = ("workspace", "work space", "iş qovluğu", "çalışma alanı")


def normalize_location_name(text: str) -> str:
    """'my Project folder' -> 'project' (for matching configured named paths)."""
    key = " ".join(text.strip().lower().split())
    for prefix in _NAME_PREFIXES:
        key = key.removeprefix(prefix)
    for suffix in _NAME_SUFFIXES:
        key = key.removesuffix(suffix)
    return key.strip()


class PathGuard:
    def __init__(self, read_dirs: list[Path], write_dirs: list[Path], *,
                 deny_dirs: list[Path] | None = None, home: Path | None = None,
                 named_paths: dict[str, Path] | None = None):
        self.read_dirs = [p.resolve() for p in read_dirs]
        self.write_dirs = [p.resolve() for p in write_dirs]
        home = (home or Path.home()).resolve()
        self.deny_dirs = [(home / d).resolve() for d in SENSITIVE_DIRS] + [
            p.resolve() for p in (deny_dirs or [])]
        self.named_paths = {k.lower(): Path(v).expanduser() for k, v in
                            (named_paths or {}).items()}
        for alias in WORKSPACE_ALIASES:
            self.named_paths.setdefault(alias, self.default_dir)

    def named(self, raw: str) -> Path | None:
        """Configured location for a bare name ('project', 'my project folder'), if any."""
        return self.named_paths.get(normalize_location_name(raw))

    def expand_named(self, raw: str) -> str:
        """Replace a leading named-location component with its configured directory.

        'project' -> <project dir>; 'project/src/a.py' -> <project dir>/src/a.py.
        Absolute, '~' and '.'-relative paths are returned unchanged. The result is still
        jail-checked by resolve().
        """
        if not raw or raw.startswith(("/", "~", ".")):
            return raw
        if (whole := self.named(raw)) is not None:
            return str(whole)
        head, sep, rest = raw.partition("/")
        if sep and (base := self.named_paths.get(head.lower())) is not None:
            return str(base / rest)
        return raw

    @property
    def default_dir(self) -> Path:
        return (self.write_dirs or self.read_dirs)[0]

    def resolve(self, raw: str | Path, mode: Literal["read", "write"] = "read") -> Path:
        s = str(raw)
        if "\x00" in s:
            raise PathNotAllowed("path contains a NUL byte")
        p = Path(self.expand_named(s)).expanduser()
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
