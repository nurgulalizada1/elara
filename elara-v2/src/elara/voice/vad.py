"""Energy-based VAD with hangover and utterance segmentation (no dependencies)."""

from __future__ import annotations

import math
from array import array

from elara.voice.interfaces import FRAME_MS


def frame_rms(frame: bytes) -> float:
    samples = array("h")
    samples.frombytes(frame[: len(frame) - len(frame) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class EnergyVAD:
    """Speech if RMS exceeds an adaptive noise floor by `ratio` (and an absolute minimum)."""

    def __init__(self, ratio: float = 3.0, min_rms: float = 300.0, floor_alpha: float = 0.05):
        self.ratio = ratio
        self.min_rms = min_rms
        self.alpha = floor_alpha
        self.noise_floor = 100.0

    def is_speech(self, frame: bytes) -> bool:
        rms = frame_rms(frame)
        speech = rms > max(self.min_rms, self.noise_floor * self.ratio)
        if not speech:  # adapt the floor only on non-speech frames
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
        return speech


class UtteranceSegmenter:
    """Groups frames into utterances: starts on speech, ends after `silence_ms` of silence."""

    def __init__(self, vad, silence_ms: int = 700, min_speech_ms: int = 300,
                 max_utterance_s: float = 30.0):
        self.vad = vad
        self.silence_frames = silence_ms // FRAME_MS
        self.min_frames = min_speech_ms // FRAME_MS
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self._buf: list[bytes] = []
        self._speech = 0
        self._silence = 0

    def push(self, frame: bytes) -> bytes | None:
        """Feed one frame; returns a complete utterance's PCM when one ends."""
        speech = self.vad.is_speech(frame)
        if not self._buf and not speech:
            return None
        self._buf.append(frame)
        if speech:
            self._speech += 1
            self._silence = 0
        else:
            self._silence += 1
        if self._silence >= self.silence_frames or len(self._buf) >= self.max_frames:
            pcm, enough = b"".join(self._buf), self._speech >= self.min_frames
            self._buf, self._speech, self._silence = [], 0, 0
            return pcm if enough else None
        return None
