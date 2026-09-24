"""`elara doctor`: actionable self-diagnostics."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from elara.config.settings import Settings
from elara.core.errors import ElaraError


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str = ""
    fix: str = ""


SYMBOL = {"ok": "✓", "warn": "⚠", "fail": "✗", "skip": "·"}


async def run_checks(settings: Settings, *, online: bool = True,
                     run_tests: bool = False) -> list[Check]:
    checks: list[Check] = []
    v = sys.version_info
    checks.append(Check("Python", "ok" if v >= (3, 12) else "fail",
                        f"{v.major}.{v.minor}.{v.micro}", "ELARA needs Python 3.12+"))

    missing = [m for m in ("fastapi", "uvicorn", "httpx", "pydantic", "pydantic_settings")
               if importlib.util.find_spec(m) is None]
    checks.append(Check("Dependencies", "fail" if missing else "ok",
                        f"missing: {', '.join(missing)}" if missing else "core packages present",
                        "pip install -e ."))
    pdf = importlib.util.find_spec("pypdf") is not None
    checks.append(Check("PDF support", "ok" if pdf else "warn",
                        "pypdf installed" if pdf else "pypdf not installed",
                        "pip install pypdf  (or: pip install -e '.[pdf]' from the source tree)"))

    checks.append(Check("Configuration", "ok", f"data dir {settings.data_dir}, language "
                                               f"{settings.default_language}"))
    container = None
    try:
        from elara.core.container import build_container
        container = build_container(settings)
        container.db.check()
        checks.append(Check("Database", "ok", f"{settings.db_path} (schema "
                                                f"v{container.db.schema_version()})"))
    except ElaraError as e:
        checks.append(Check("Database", "fail", str(e), "check ELARA_DATA_DIR permissions"))
    except OSError as e:
        checks.append(Check("Database", "fail", str(e), "check ELARA_DATA_DIR permissions"))

    provider = settings.resolved_provider()
    if container is None or not container.llm.available:
        checks.append(Check("LLM provider", "warn", f"none configured ({provider})",
                            "export ANTHROPIC_API_KEY=... (or OPENAI_API_KEY); deterministic "
                            "features work without it"))
    elif online:
        try:
            status = await container.llm.provider.check()  # type: ignore[union-attr]
            checks.append(Check(f"LLM provider ({provider})", "ok",
                                f"{status}; fast={settings.model_for('fast')}, "
                                f"strong={settings.model_for('strong')}"))
        except ElaraError as e:
            checks.append(Check(f"LLM provider ({provider})", "fail", str(e),
                                "verify the API key and network access"))
    else:
        checks.append(Check(f"LLM provider ({provider})", "skip", "offline mode: key present"))

    if container:
        n_mem = len(container.memory.store.list(limit=100_000))
        checks.append(Check("Memory", "ok", f"{n_mem} active memories, FTS5 search"))
        exposed = sum(t.exposed_to_llm for t in container.registry.all())
        checks.append(Check("Tools", "ok", f"{len(container.registry)} registered "
                                           f"({exposed} available to the model)"))
        ws = settings.write_dirs
        missing_ws = [str(p) for p in ws if not p.exists()]
        checks.append(Check("Permissions", "warn" if missing_ws else "ok",
                            f"read: {', '.join(map(str, settings.read_dirs))}; write: "
                            f"{', '.join(map(str, ws))}; code execution "
                            f"{'ON' if settings.enable_code_execution else 'off'}",
                            f"mkdir -p {' '.join(missing_ws)}" if missing_ws else ""))
        if online:
            checks.extend(await _research_checks(container))
        await container.aclose()

    checks.extend(_voice_checks(settings))
    if run_tests:
        checks.append(_tests())
    else:
        checks.append(Check("Test suite", "skip", "run `elara doctor --tests` to execute"))
    return checks


async def _research_checks(container) -> list[Check]:
    async def probe(source) -> Check:
        if not source.health_url:
            return Check(f"{source.name} API", "skip", "no cheap health endpoint (POST-only)")
        try:
            r = await container.http.get(source.health_url, timeout=8,
                                         headers={"user-agent": "ELARA-doctor"})
            if r.status_code < 400:
                return Check(f"{source.name} API", "ok", f"HTTP {r.status_code}")
            hint = ""
            if r.status_code == 429 and source.name == "semantic_scholar":
                hint = ("rate limited — set ELARA_SEMANTIC_SCHOLAR_API_KEY; research only calls "
                        "Semantic Scholar when PubMed/Europe PMC return too few results")
            elif r.status_code == 429:
                hint = "rate limited — try later or configure an API key for this source"
            return Check(f"{source.name} API", "warn", f"HTTP {r.status_code}", hint)
        except httpx.HTTPError as e:
            return Check(f"{source.name} API", "warn", f"unreachable ({type(e).__name__})",
                         "check network / proxy; research will skip this source")
    return list(await asyncio.gather(*(probe(s) for s in container.research.sources.values())))


def _voice_checks(settings: Settings) -> list[Check]:
    out = []
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    out.append(Check("Voice: TTS (espeak-ng)", "ok" if espeak else "warn",
                     espeak or "not installed", "sudo apt install espeak-ng"))
    fw = importlib.util.find_spec("faster_whisper") is not None
    out.append(Check("Voice: STT (faster-whisper)", "ok" if fw else "warn",
                     "installed" if fw else "not installed", "pip install faster-whisper"))
    oww = importlib.util.find_spec("openwakeword") is not None
    out.append(Check("Voice: wake word (openWakeWord)", "ok" if oww else "warn",
                     "installed" if oww else "not installed (and a 'hey elara' model is needed)",
                     "pip install openwakeword; train/provide a 'hey_elara' model"))
    return out


def _tests() -> Check:
    root = Path(__file__).resolve().parents[3]
    if not (root / "tests").is_dir():
        return Check("Test suite", "skip", "tests directory not found (installed package?)")
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "--no-header"],
                              cwd=root, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return Check("Test suite", "fail", str(e))
    last = (proc.stdout.strip().splitlines() or ["no output"])[-1]
    return Check("Test suite", "ok" if proc.returncode == 0 else "fail", last,
                 "" if proc.returncode == 0 else "run pytest -x to see the failure")


def render(checks: list[Check], color: bool = True) -> str:
    colors = {"ok": "\033[32m", "warn": "\033[33m", "fail": "\033[31m", "skip": "\033[2m"}
    lines = ["ELARA Doctor", ""]
    for ch in checks:
        sym = SYMBOL[ch.status]
        if color:
            sym = f"{colors[ch.status]}{sym}\033[0m"
        lines.append(f"{sym} {ch.name}: {ch.detail}")
        if ch.fix and ch.status in ("warn", "fail"):
            lines.append(f"    → {ch.fix}")
    fails = sum(ch.status == "fail" for ch in checks)
    warns = sum(ch.status == "warn" for ch in checks)
    lines += ["", f"{fails} failure(s), {warns} warning(s)"]
    return "\n".join(lines)
