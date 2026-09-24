"""Energy-based VAD with hangover and utterance segmentation (no dependencies)."""

from __future__ import annotations

import math
from array import array
from collections import deque

from elara.voice.interfaces import FRAME_MS


def frame_levels(frame: bytes) -> tuple[float, float]:
    """``(mean, ac_rms)`` of one int16 frame.

    ``ac_rms`` is the RMS of the mean-centred samples, so a constant (DC) offset from the
    capture hardware does not count as sound; within a 30 ms frame this removes only content
    below ~33 Hz, far under speech. ``mean`` is returned for diagnostics only.
    """
    samples = array("h")
    samples.frombytes(frame[: len(frame) - len(frame) % 2])
    n = len(samples)
    if not n:
        return 0.0, 0.0
    total = sum(samples)
    power = sum(s * s for s in samples)
    # n*sum(s^2) - sum(s)^2 == n^2 * variance, exact in integers
    return total / n, math.sqrt(max(0, n * power - total * total)) / n


def frame_rms(frame: bytes) -> float:
    """DC-offset invariant (AC) RMS of one int16 frame."""
    return frame_levels(frame)[1]


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

    All levels are AC RMS (``frame_levels``), so a DC offset never looks like sound.

    - Noise floor: a low percentile (default 15th) of the AC RMS of recent non-speech frames
      in a rolling window (default 40 frames = 1.2 s). A percentile is not raised by a
      syllable, so nothing assumes the start of listening is quiet. Frames confirmed as
      speech are excluded, so a long utterance does not ratchet the floor up.
    - Warm-up: until ``calibration_frames`` non-speech frames exist the floor is unknown and
      speech may start on the absolute ``min_rms`` alone, so talking right away still works.
      If speech runs a whole window without any quiet frame, those levels seed the floor
      (a loud steady room that tripped the absolute minimum then ends instead of hanging).
    - Hysteresis: speech starts above ``max(min_rms, floor * start_ratio)`` for
      ``start_frames`` consecutive frames (clicks are shorter), and continues until the
      smoothed level drops below ``floor * stop_ratio``.
    - End decisions use a short moving average, so one noisy frame does not reset silence.
    """

    def __init__(self, *, min_rms: float = 300.0, start_ratio: float = 3.0,
                 stop_ratio: float = 2.0, calibration_frames: int = 8, start_frames: int = 3,
                 smoothing_frames: int = 3, floor_window_frames: int = 40,
                 floor_percentile: float = 0.15):
        if stop_ratio > start_ratio:
            raise ValueError("stop_ratio must not exceed start_ratio")
        self.min_rms = min_rms
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self.calibration_frames = max(1, calibration_frames)
        self.start_frames = max(1, start_frames)
        self.floor_percentile = floor_percentile
        window = max(floor_window_frames, self.calibration_frames)
        self._quiet: deque[float] = deque(maxlen=window)   # AC RMS of non-speech frames
        self._blind: deque[float] = deque(maxlen=window)   # speech levels while floor unknown
        self._means: deque[float] = deque(maxlen=window)   # frame means (diagnostics only)
        self._recent: deque[float] = deque(maxlen=max(1, smoothing_frames))
        self.in_speech = False
        self._above = 0

    @property
    def noise_floor(self) -> float | None:
        if len(self._quiet) < self.calibration_frames:
            return None
        ordered = sorted(self._quiet)
        return max(1.0, ordered[int(len(ordered) * self.floor_percentile)])

    @property
    def calibrating(self) -> bool:
        return self.noise_floor is None

    @property
    def start_threshold(self) -> float:
        floor = self.noise_floor
        return self.min_rms if floor is None else max(self.min_rms, floor * self.start_ratio)

    @property
    def stop_threshold(self) -> float:
        return max(self.min_rms * self.stop_ratio / self.start_ratio,
                   (self.noise_floor or 0.0) * self.stop_ratio)

    @property
    def dc_offset(self) -> float | None:
        """Mean sample value over the recent window (diagnostic; never used for decisions)."""
        return sum(self._means) / len(self._means) if self._means else None

    def update(self, frame: bytes) -> str:
        """Feed one frame; returns 'calibrating', 'silence', 'start', 'speech' or 'pause'."""
        mean, rms = frame_levels(frame)
        self._means.append(mean)
        self._recent.append(rms)
        if not self.in_speech:
            warming_up = self.calibrating
            above = rms > self.start_threshold
            self._quiet.append(rms)
            if not above:
                self._above = 0
                return "calibrating" if warming_up else "silence"
            self._above += 1
            if self._above < self.start_frames:
                return "calibrating" if warming_up else "silence"
            self.in_speech = True
            for _ in range(min(self.start_frames, len(self._quiet))):
                self._quiet.pop()  # the confirmed onset is speech, not background
            return "start"
        if self.calibrating:
            self._blind.append(rms)
            if len(self._blind) == self._blind.maxlen:  # a window without a quiet frame
                self._quiet.extend(self._blind)
                self._blind.clear()
        smoothed = sum(self._recent) / len(self._recent)
        if smoothed >= self.stop_threshold:
            return "speech"
        if rms < self.stop_threshold:  # keep tracking the background during pauses
            self._quiet.append(rms)
        return "pause"
