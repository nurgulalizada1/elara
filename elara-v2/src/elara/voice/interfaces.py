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
    text: str
    language: str | None = None
    confidence: float | None = None      # 0..1 (e.g. exp(avg_logprob))
    no_speech_prob: float | None = None
    duration_s: float = 0.0


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
