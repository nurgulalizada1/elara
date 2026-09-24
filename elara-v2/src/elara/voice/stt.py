"""Speech-to-text: faster-whisper on CPU, English-only production path.

The model is loaded once (``load()``) and reused for every utterance. Language is forced
to the configured language ("en"); requests for any other language are refused so an
Azerbaijani (or other) path cannot be enabled by accident. faster-whisper is imported
lazily, so the rest of ELARA works without the voice extra installed.
"""

from __future__ import annotations

import time

from elara.core.errors import ElaraError
from elara.core.logging import get_logger
from elara.voice.config import SAMPLE_RATE, VoiceConfig
from elara.voice.interfaces import Transcript

log = get_logger(__name__)


class STTUnavailable(ElaraError):
    pass


class FasterWhisperSTT:
    """Local Whisper via faster-whisper (CTranslate2). Transcription only — never used
    for wake-word detection."""

    name = "faster-whisper"

    def __init__(self, config: VoiceConfig | None = None, model_factory=None):
        self.config = config or VoiceConfig()
        self._factory = model_factory
        self._model = None
        self.load_time_s: float | None = None

    @property
    def language(self) -> str:
        return self.config.language

    def load(self) -> float:
        """Load the model once; returns load time in seconds (0 if already loaded)."""
        if self._model is not None:
            return 0.0
        factory = self._factory
        if factory is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise STTUnavailable("faster-whisper is not installed "
                                     "(pip install 'elara[voice]')") from e
            factory = WhisperModel
        c = self.config
        t0 = time.perf_counter()
        try:
            self._model = factory(c.model, device=c.device, compute_type=c.compute_type,
                                  cpu_threads=c.cpu_threads, num_workers=1,
                                  download_root=str(c.model_dir) if c.model_dir else None,
                                  local_files_only=c.local_files_only)
        except Exception as e:  # download blocked, corrupt cache, unknown model name
            raise STTUnavailable(f"cannot load Whisper model '{c.model}': {e}") from e
        self.load_time_s = round(time.perf_counter() - t0, 2)
        log.info("voice.stt_loaded", extra={"model": c.model, "compute_type": c.compute_type,
                                            "cpu_threads": c.cpu_threads,
                                            "load_s": self.load_time_s})
        return self.load_time_s

    def transcribe(self, pcm: bytes, language: str | None = None) -> Transcript:
        """Transcribe 16 kHz mono int16 PCM. Failures are returned, not raised."""
        if language is not None and language != self.language:
            raise ValueError(f"voice STT is configured for '{self.language}' only; "
                             f"'{language}' is not a production voice language")
        import numpy as np

        self.load()
        duration = len(pcm) / (SAMPLE_RATE * 2)
        audio = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
        audio = audio.astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        try:
            segments, info = self._model.transcribe(
                audio, language=self.language, beam_size=self.config.beam_size,
                vad_filter=False, condition_on_previous_text=False)
            segs = list(segments)  # decoding happens while iterating
        except Exception as e:
            log.warning("voice.stt_failed", extra={"error": type(e).__name__})
            return Transcript(text="", language=self.language, duration_s=round(duration, 3),
                              transcription_time_s=round(time.perf_counter() - t0, 3),
                              error=f"{type(e).__name__}: {e}")
        elapsed = round(time.perf_counter() - t0, 3)
        return Transcript(
            text=" ".join(s.text.strip() for s in segs).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration_s=round(duration, 3),
            transcription_time_s=elapsed,
            avg_logprob=round(sum(s.avg_logprob for s in segs) / len(segs), 3) if segs else None,
            no_speech_prob=round(max(s.no_speech_prob for s in segs), 3) if segs else None,
            compression_ratio=round(max(s.compression_ratio for s in segs), 3) if segs else None,
        )
