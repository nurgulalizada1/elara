"""TTS providers. Pluggable; selected by ELARA_TTS_PROVIDER with per-language voices."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable

from elara.core.errors import ElaraError


class TTSUnavailable(ElaraError):
    pass


class EspeakTTS:
    """espeak-ng: offline, supports Azerbaijani (az), English and Turkish. Robotic but real."""

    name = "espeak"

    def __init__(self, voices: dict[str, str] | None = None, rate_wpm: int = 165,
                 binary: str | None = None):
        self.voices = voices or {"az": "az", "en": "en-us", "tr": "tr"}
        self.rate = rate_wpm
        self.binary = binary or shutil.which("espeak-ng") or shutil.which("espeak")

    def languages(self) -> list[str]:
        return list(self.voices)

    def synthesize(self, text: str, language: str) -> bytes:
        if not self.binary:
            raise TTSUnavailable("espeak-ng is not installed (sudo apt install espeak-ng)")
        voice = self.voices.get(language) or self.voices.get("en", "en")
        try:
            proc = subprocess.run([self.binary, "-v", voice, "-s", str(self.rate), "--stdout",
                                   text[:5000]], capture_output=True, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise TTSUnavailable(f"espeak failed: {e}") from e
        if proc.returncode != 0 or not proc.stdout.startswith(b"RIFF"):
            raise TTSUnavailable(f"espeak failed: {proc.stderr.decode(errors='replace')[:200]}")
        return proc.stdout


TTS_PROVIDERS: dict[str, Callable[..., object]] = {"espeak": EspeakTTS}


def build_tts(name: str, voices: dict[str, str]):
    try:
        factory = TTS_PROVIDERS[name]
    except KeyError as e:
        raise TTSUnavailable(f"unknown TTS provider '{name}' (available: "
                             f"{', '.join(TTS_PROVIDERS)})") from e
    return factory(voices=voices)
