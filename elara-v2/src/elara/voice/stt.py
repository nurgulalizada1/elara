"""Speech-to-text adapters. faster-whisper is optional and imported lazily."""

from __future__ import annotations

import math
from array import array

from elara.core.errors import ElaraError
from elara.voice.interfaces import SAMPLE_RATE, Transcript


class STTUnavailable(ElaraError):
    pass


class FasterWhisperSTT:
    """Local Whisper via faster-whisper (CTranslate2). Transcription only — never used
    for wake-word detection."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "small", device: str = "auto",
                 compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise STTUnavailable("faster-whisper is not installed "
                                 "(pip install faster-whisper)") from e
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm: bytes, language: str | None = None) -> Transcript:
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
        audio = [s / 32768.0 for s in samples]
        try:
            import numpy as np  # faster-whisper depends on numpy
            audio = np.asarray(audio, dtype=np.float32)
        except ImportError:
            pass
        segments, info = self._model.transcribe(audio, language=language, vad_filter=False,
                                                beam_size=5)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        logprob = sum(s.avg_logprob for s in segs) / len(segs) if segs else -10.0
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        return Transcript(text=text, language=info.language, confidence=math.exp(logprob),
                          no_speech_prob=no_speech, duration_s=len(samples) / SAMPLE_RATE)
