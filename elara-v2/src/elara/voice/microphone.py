"""Microphone capture: one bounded utterance at a time, 16 kHz mono int16, local only.

PortAudio (via sounddevice, on PipeWire) delivers 30 ms frames to a callback that only
enqueues them; the consumer blocks on the queue with a timeout (no busy loop). Capture
stops at end of speech, at the utterance cap, or when no speech starts in time. Audio is
returned in memory and never written to disk or sent anywhere by this module.
"""

from __future__ import annotations

import queue
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from elara.core.errors import ElaraError
from elara.core.logging import get_logger
from elara.voice.config import CHANNELS, FRAME_MS, SAMPLE_RATE, VoiceConfig
from elara.voice.vad import SpeechEndpointer, rms_dbfs

log = get_logger(__name__)


class MicrophoneError(ElaraError):
    """Capture failed (no PortAudio, unknown device, device error)."""


@dataclass
class Utterance:
    pcm: bytes                 # 16 kHz mono int16 little-endian
    duration_s: float
    capture_s: float           # wall time from start of listening to end of utterance
    reason: str                # end_of_speech | max_duration | no_speech
    overflows: int = 0
    # Diagnostics (levels only, never audio): what the endpointer decided with.
    noise_floor_dbfs: float | None = None
    start_threshold_dbfs: float | None = None
    stop_threshold_dbfs: float | None = None
    speech_ms: int = 0
    trailing_silence_ms: int = 0


# A stream factory returns an object with start()/stop()/close() that calls
# callback(indata_bytes) for every frame. Injectable for hardware-free tests.
StreamFactory = Callable[[Callable[[bytes], None]], Any]


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio shared library missing
        raise MicrophoneError("PortAudio is not available (sudo apt install libportaudio2)") from e
    except ImportError as e:
        raise MicrophoneError("sounddevice is not installed (pip install 'elara[voice]')") from e
    return sd


def list_input_devices() -> list[dict]:
    sd = _sounddevice()
    default_in = sd.default.device[0]
    return [{"index": i, "name": d["name"], "channels": d["max_input_channels"],
             "default_samplerate": d["default_samplerate"], "is_default": i == default_in}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


class Microphone:
    def __init__(self, config: VoiceConfig, stream_factory: StreamFactory | None = None):
        self.config = config
        self._factory = stream_factory or self._sounddevice_stream

    def _device(self):
        d = self.config.input_device
        return int(d) if isinstance(d, str) and d.isdigit() else d

    def _sounddevice_stream(self, on_frame: Callable[[bytes], None]):
        sd = _sounddevice()
        overflow = {"n": 0}

        def callback(indata, frames, time_info, status):  # PortAudio thread: enqueue only
            if status.input_overflow:
                overflow["n"] += 1
            on_frame(bytes(indata))

        try:
            stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                                       blocksize=self.config.frame_samples,
                                       device=self._device(), callback=callback)
        except (ValueError, sd.PortAudioError) as e:
            raise MicrophoneError(f"cannot open input device {self._device()!r}: {e}") from e
        stream.overflow_counter = overflow
        return stream

    def listen(self) -> Utterance:
        """Block until one utterance is captured (or no speech starts in time)."""
        cfg = self.config
        frames: queue.Queue[bytes] = queue.Queue(maxsize=int(10_000 / FRAME_MS))
        stream = self._factory(lambda b: frames.put_nowait(b) if not frames.full() else None)
        ep = SpeechEndpointer(min_rms=cfg.vad_min_rms, start_ratio=cfg.vad_ratio,
                              stop_ratio=min(cfg.vad_stop_ratio, cfg.vad_ratio),
                              calibration_frames=cfg.calibration_ms // FRAME_MS,
                              start_frames=cfg.speech_start_ms // FRAME_MS,
                              smoothing_frames=cfg.vad_smoothing_ms // FRAME_MS)
        preroll: deque[bytes] = deque(maxlen=max(1, cfg.preroll_ms // FRAME_MS))
        captured: list[bytes] = []
        speech_frames = silence_frames = 0
        end_silence = cfg.end_silence_ms // FRAME_MS
        max_frames = int(cfg.max_utterance_s * 1000 / FRAME_MS)
        t0 = time.monotonic()
        reason = "no_speech"
        try:
            stream.start()
        except Exception as e:
            raise MicrophoneError(f"cannot start microphone: {e}") from e
        try:
            while True:
                if not captured and time.monotonic() - t0 > cfg.start_timeout_s:
                    break
                try:
                    frame = frames.get(timeout=0.5)
                except queue.Empty:
                    if not captured:
                        continue
                    reason = "end_of_speech"  # stream stalled mid-utterance: stop cleanly
                    break
                state = ep.update(frame)
                if not captured:
                    preroll.append(frame)
                    if state == "start":
                        captured.extend(preroll)  # pre-roll keeps the first syllable
                        speech_frames = ep.start_frames
                    continue
                captured.append(frame)
                if state == "speech":
                    speech_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1
                if silence_frames >= end_silence:
                    reason = "end_of_speech"
                    break
                if len(captured) >= max_frames:
                    reason = "max_duration"
                    break
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:  # closing must never mask the capture result
                log.warning("voice.mic_close_failed")
        if captured and speech_frames * FRAME_MS < cfg.min_speech_ms:
            captured, reason = [], "no_speech"  # a click or bump, not speech
        pcm = b"".join(captured)
        overflows = getattr(stream, "overflow_counter", {}).get("n", 0)
        floor = ep.noise_floor
        return Utterance(pcm=pcm, duration_s=round(len(pcm) / (SAMPLE_RATE * 2), 3),
                         capture_s=round(time.monotonic() - t0, 3), reason=reason,
                         overflows=overflows,
                         noise_floor_dbfs=rms_dbfs(floor) if floor else None,
                         start_threshold_dbfs=rms_dbfs(ep.start_threshold) if floor else None,
                         stop_threshold_dbfs=rms_dbfs(ep.stop_threshold) if floor else None,
                         speech_ms=speech_frames * FRAME_MS if pcm else 0,
                         trailing_silence_ms=silence_frames * FRAME_MS if pcm else 0)
