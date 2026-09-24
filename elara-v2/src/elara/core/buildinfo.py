"""Identify exactly which ELARA code is running (version, git commit, location).

Real-machine validation once ran a stale install while newer code existed; this makes
that visible in `elara --version`, `elara doctor`, `/health` and every `ask --json` reply.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import elara


@lru_cache(maxsize=1)
def build_info() -> dict[str, str | None]:
    pkg = Path(elara.__file__).resolve().parent
    commit = None
    git = shutil.which("git")
    if git:
        try:
            out = subprocess.run([git, "-C", str(pkg), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=3, check=False)
            if out.returncode == 0:
                commit = out.stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"version": elara.__version__, "commit": commit, "code_path": str(pkg)}


def build_label() -> str:
    b = build_info()
    commit = f"commit {b['commit']}" if b["commit"] else "no git checkout"
    return f"elara {b['version']} ({commit}; code: {b['code_path']})"
