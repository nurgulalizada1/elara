"""Voice configuration (environment prefix ``ELARA_VOICE_``).

Production voice input is English-only by decision (Azerbaijani STT measured at ~60% WER
with the CPU model; see bench/voice). The language field therefore only accepts "en";
Azerbaijani STT stays in the benchmark harness as an experimental capability. The text
interface and ElaraCore remain fully multilingual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

SAMPLE_RATE = 16_000   # Hz, fixed: Whisper's native rate
CHANNELS = 1           # mono
SAMPLE_WIDTH = 2       # bytes: 16-bit signed PCM
FRAME_MS = 30


class VoiceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ELARA_VOICE_", env_file=".env",
                                      extra="ignore")

    enabled: bool = True
    # --- speech-to-text (faster-whisper) ---
    language: Literal["en"] = "en"          # English-only production path
    model: str = "small"
    device: Literal["cpu"] = "cpu"          # no CUDA/ROCm on the target machine
    compute_type: str = "int8"
    cpu_threads: int = Field(default=8, ge=1, le=64)
    beam_size: int = Field(default=5, ge=1, le=10)
    model_dir: Path | None = None           # None = Hugging Face cache default
    local_files_only: bool = False          # True = never download the model
    # --- capture ---
    input_device: str | None = None         # sounddevice index or name; None = default
    start_timeout_s: float = Field(default=8.0, gt=0)     # wait this long for speech
    max_utterance_s: float = Field(default=12.0, gt=0)    # hard cap per utterance
    min_speech_ms: int = Field(default=250, ge=0)         # shorter bursts are noise
    end_silence_ms: int = Field(default=800, ge=90)       # silence that ends an utterance
    preroll_ms: int = Field(default=300, ge=0)            # audio kept from before speech
    vad_min_rms: float = Field(default=300.0, gt=0)       # int16 RMS (~ -40 dBFS)
    vad_ratio: float = Field(default=3.0, gt=1)           # speech = ratio x noise floor
    # --- speech output ---
    tts_enabled: bool = True
    tts_voice: str = "en-us"
    tts_rate_wpm: int = Field(default=170, ge=80, le=400)

    @property
    def frame_samples(self) -> int:
        return SAMPLE_RATE * FRAME_MS // 1000
