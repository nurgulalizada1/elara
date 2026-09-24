"""TTS providers. Pluggable; selected by ELARA_TTS_PROVIDER with per-language voices."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from typing import Protocol

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


# --------------------------------------------------------------- spoken responses ----
class Speaker(Protocol):
    """Replaceable speech output used by the voice channel. English-only for now."""

    name: str

    def available(self) -> bool: ...

    def speak(self, text: str) -> float:
        """Speak `text`; return seconds spent (synthesis + playback)."""
        ...


def play_wav(wav: bytes) -> None:
    """Play WAV bytes on the default output device via sounddevice (blocking)."""
    import io
    import wave

    import numpy as np

    try:
        import sounddevice as sd
    except (OSError, ImportError) as e:
        raise TTSUnavailable(f"audio output unavailable: {e}") from e
    with wave.open(io.BytesIO(wav), "rb") as w:
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    sd.play(data.reshape(-1, channels) if channels > 1 else data, samplerate=rate)
    sd.wait()


class EspeakSpeaker:
    """Local fallback: espeak-ng English voice, played through sounddevice."""

    name = "espeak"

    def __init__(self, voice: str = "en-us", rate_wpm: int = 170, binary: str | None = None,
                 player: Callable[[bytes], None] = play_wav, max_chars: int = 600):
        self._tts = EspeakTTS(voices={"en": voice}, rate_wpm=rate_wpm, binary=binary)
        self._player = player
        self.max_chars = max_chars

    def available(self) -> bool:
        return bool(self._tts.binary)

    def speak(self, text: str) -> float:
        import time

        t0 = time.perf_counter()
        spoken = text.strip()
        if len(spoken) > self.max_chars:  # long answers (e.g. research) stay on screen
            spoken = spoken[: self.max_chars].rsplit(" ", 1)[0] + " …"
        if spoken:
            self._player(self._tts.synthesize(spoken, "en"))
        return round(time.perf_counter() - t0, 3)
