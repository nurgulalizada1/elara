"""Voice component contracts. Wake-word, VAD, STT and TTS are deliberately separate.

Audio convention: mono, 16-bit signed little-endian PCM, 16 kHz, frames of 30 ms
(480 samples = 960 bytes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2


@dataclass
class Transcript:
    """STT output. Only fields the engine actually reports are filled; None = not available."""

    text: str
    language: str | None = None
    confidence: float | None = None      # 0..1, only for engines that report one
    no_speech_prob: float | None = None  # faster-whisper: max over segments
    duration_s: float = 0.0              # audio duration
    language_probability: float | None = None  # 1.0 when the language was forced
    avg_logprob: float | None = None     # faster-whisper: mean over segments
    compression_ratio: float | None = None  # faster-whisper: max over segments
    transcription_time_s: float = 0.0
    error: str | None = None             # set when transcription failed

    @property
    def real_time_factor(self) -> float | None:
        return round(self.transcription_time_s / self.duration_s, 3) if self.duration_s else None


@runtime_checkable
class VoiceActivityDetector(Protocol):
    def is_speech(self, frame: bytes) -> bool: ...


@runtime_checkable
class WakeWordDetector(Protocol):
    """Listens continuously for the wake phrase ("Hey ELARA"). Never transcribes."""

    phrase: str

    def detect(self, frame: bytes) -> bool: ...


@runtime_checkable
class SpeechToText(Protocol):
    name: str

    def transcribe(self, pcm: bytes, language: str | None = None) -> Transcript: ...


@runtime_checkable
class TextToSpeech(Protocol):
    name: str

    def languages(self) -> list[str]: ...

    def synthesize(self, text: str, language: str) -> bytes:
        """Return WAV bytes."""
        ...
