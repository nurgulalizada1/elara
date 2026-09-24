"""Controlled system tools. There is deliberately NO generic shell tool."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from typing import ClassVar

from pydantic import BaseModel, Field

from elara.core.errors import ToolError
from elara.security.paths import PathGuard, normalize_location_name
from elara.security.untrusted import Trust
from elara.tools.base import Permission, Tool, ToolContext


class OpenInput(BaseModel):
    path: str = Field(description="Path, or a configured name such as 'project'")


class OpenOutput(BaseModel):
    path: str
    opened_with: str


class OpenPathTool(Tool):
    name: ClassVar[str] = "open_path"
    description: ClassVar[str] = ("Open a folder (or file) in the desktop's default application "
                                  "via xdg-open. Accepts configured names like 'project'. Opening "
                                  "files needs confirmation; opening folders does not.")
    Input = OpenInput
    Output = OpenOutput
    permission = Permission.SYSTEM
    category = "system"
    timeout_s = 10.0

    def __init__(self, guard: PathGuard, named_paths: dict | None = None,
                 opener: str | None = None):
        self.guard = guard
        self.named = named_paths or {}
        self.opener = opener

    def _target(self, raw: str):
        named = self.named.get(normalize_location_name(raw)) or self.guard.named(raw)
        return self.guard.resolve(str(named) if named else raw, "read")

    def permission_for(self, args: OpenInput) -> Permission:
        try:
            p = self._target(args.path)
        except Exception:
            return Permission.SYSTEM
        # A folder opens a file manager; a file may be an executable/.desktop launcher.
        return Permission.LOW_RISK_WRITE if p.is_dir() else Permission.SYSTEM

    async def run(self, args: OpenInput, ctx: ToolContext) -> OpenOutput:
        p = self._target(args.path)
        if not p.exists():
            raise ToolError(f"does not exist: {p}")
        opener = self.opener or shutil.which("xdg-open")
        if not opener:
            raise ToolError("xdg-open is not available (no desktop session?)")
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") \
                and self.opener is None:
            raise ToolError("no graphical session detected (DISPLAY/WAYLAND_DISPLAY unset)")
        proc = await asyncio.create_subprocess_exec(
            opener, str(p), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=8)
        except TimeoutError:
            return OpenOutput(path=str(p), opened_with=opener)  # launcher still running: fine
        if proc.returncode != 0:
            raise ToolError(f"{opener} failed: {err.decode(errors='replace')[:200]}")
        return OpenOutput(path=str(p), opened_with=opener)


class PythonInput(BaseModel):
    code: str = Field(max_length=20_000, description="Python source to run")
    timeout_s: float = Field(default=10, ge=1, le=60)


class PythonOutput(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool


def _limits():  # runs in the child before exec
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    os.setsid()


class RunPythonTool(Tool):
    name: ClassVar[str] = "run_python"
    description: ClassVar[str] = ("Run a short Python script in an isolated subprocess (temp "
                                  "dir, no inherited environment/secrets, memory/CPU limits). "
                                  "Always requires user confirmation.")
    Input = PythonInput
    Output = PythonOutput
    permission = Permission.SYSTEM
    category = "system"
    timeout_s = 70.0
    output_trust = Trust.UNTRUSTED
    safety_notes = ("Disabled unless ELARA_ENABLE_CODE_EXECUTION=true. Not a full sandbox: "
                    "the process runs as your user and can reach the network.")

    async def run(self, args: PythonInput, ctx: ToolContext) -> PythonOutput:
        with tempfile.TemporaryDirectory(prefix="elara-py-") as tmp:
            env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONIOENCODING": "utf-8",
                   "LANG": "C.UTF-8"}
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", "-c", args.code, cwd=tmp, env=env,
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, preexec_fn=_limits)  # noqa: PLW1509
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=args.timeout_s)
                timed_out = False
            except TimeoutError:
                proc.kill()
                out, err = await proc.communicate()
                timed_out = True
        return PythonOutput(exit_code=proc.returncode if proc.returncode is not None else -9,
                            stdout=out.decode(errors="replace")[-10_000:],
                            stderr=err.decode(errors="replace")[-5_000:], timed_out=timed_out)
