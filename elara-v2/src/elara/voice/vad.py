"""Energy-based VAD with hangover and utterance segmentation (no dependencies)."""

from __future__ import annotations

import math
from array import array
from collections import deque

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


def rms_dbfs(rms: float) -> float:
    return round(20 * math.log10(rms / 32768), 1) if rms > 0 else -120.0


class SpeechEndpointer:
    """Start/end-of-speech decisions for one utterance (used by the production microphone).

    - The noise floor is calibrated from the first frames (a low percentile, so an early
      syllable does not inflate it) and then tracks quiet frames slowly. Unlike EnergyVAD it
      does not start at a fixed guess, so a laptop mic whose background is above the absolute
      minimum no longer classifies *everything* as speech.
    - Hysteresis: speech starts above ``floor * start_ratio`` (and ``min_rms``) for
      ``start_frames`` consecutive frames (clicks are shorter), and continues until the
      smoothed level drops below ``floor * stop_ratio``.
    - End decisions use a short moving average, so one noisy frame does not reset silence.
    """

    def __init__(self, *, min_rms: float = 300.0, start_ratio: float = 3.0,
                 stop_ratio: float = 2.0, calibration_frames: int = 8, start_frames: int = 3,
                 smoothing_frames: int = 3, floor_alpha: float = 0.05):
        if stop_ratio > start_ratio:
            raise ValueError("stop_ratio must not exceed start_ratio")
        self.min_rms = min_rms
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self.calibration_frames = calibration_frames
        self.start_frames = max(1, start_frames)
        self.alpha = floor_alpha
        self._calib: list[float] = []
        self._recent: deque[float] = deque(maxlen=max(1, smoothing_frames))
        self.noise_floor: float | None = None
        self.in_speech = False
        self._above = 0

    @property
    def calibrating(self) -> bool:
        return self.noise_floor is None

    @property
    def start_threshold(self) -> float:
        return max(self.min_rms, (self.noise_floor or 0.0) * self.start_ratio)

    @property
    def stop_threshold(self) -> float:
        return max(self.min_rms * self.stop_ratio / self.start_ratio,
                   (self.noise_floor or 0.0) * self.stop_ratio)

    def _finish_calibration(self) -> None:
        ordered = sorted(self._calib)
        self.noise_floor = max(1.0, ordered[len(ordered) // 5])  # 20th percentile

    def update(self, frame: bytes) -> str:
        """Feed one frame; returns 'calibrating', 'silence', 'start', 'speech' or 'pause'."""
        rms = frame_rms(frame)
        self._recent.append(rms)
        if self.calibrating:
            self._calib.append(rms)
            if len(self._calib) >= self.calibration_frames:
                self._finish_calibration()
            return "calibrating"
        if not self.in_speech:
            if rms > self.start_threshold:
                self._above += 1
                if self._above >= self.start_frames:
                    self.in_speech = True
                    return "start"
                return "silence"
            self._above = 0
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
            return "silence"
        smoothed = sum(self._recent) / len(self._recent)
        if smoothed >= self.stop_threshold:
            return "speech"
        if rms < self.stop_threshold:  # keep tracking slow changes in the background
            self.noise_floor = (1 - self.alpha) * self.noise_floor + self.alpha * rms
        return "pause"
