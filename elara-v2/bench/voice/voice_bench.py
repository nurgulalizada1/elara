#!/usr/bin/env python3
"""Isolated CPU voice benchmark for ELARA (engineering tool, NOT production code).

Nothing here is imported by the `elara` package. Recordings are written to local disk only
and are never uploaded or played back automatically. The only network access is the
one-time Whisper model download from Hugging Face when a model is first used (audio is
never sent anywhere).

Subcommands:
  devices                         list capture devices and the default input
  mic-check  [--seconds 5]        record, verify the WAV, report RMS/peak
  record     --seconds N --out F  record a clip (16 kHz, mono, 16-bit PCM)
  transcribe WAV --model small    transcribe with faster-whisper on CPU (int8, language=az)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
import wave
from array import array
from datetime import datetime
from pathlib import Path

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes -> 16-bit PCM
HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / "recordings"
MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# --------------------------------------------------------------------- audio helpers --
def _sd():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio shared library missing
        sys.exit(f"sounddevice cannot load PortAudio ({e}).\n"
                 "Install the system library:  sudo apt install libportaudio2")
    return sd


def list_devices() -> dict:
    sd = _sd()
    default_in = sd.default.device[0]
    hostapis = sd.query_hostapis()
    inputs = []
    for idx, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            inputs.append({"index": idx, "name": d["name"],
                           "hostapi": hostapis[d["hostapi"]]["name"],
                           "max_input_channels": d["max_input_channels"],
                           "default_samplerate": d["default_samplerate"],
                           "is_default": idx == default_in})
    return {"default_input": default_in, "inputs": inputs}


def record(seconds: float, out: Path, device: str | int | None = None) -> Path:
    """Record `seconds` of 16 kHz mono int16 audio to `out` (local file only)."""
    if not 0.5 <= seconds <= 120:
        raise ValueError("duration must be between 0.5 and 120 seconds")
    sd = _sd()
    dev = int(device) if isinstance(device, str) and device.isdigit() else device
    frames = int(seconds * SAMPLE_RATE)
    print(f"Recording {seconds:.1f}s from device {dev if dev is not None else 'default'} "
          f"... speak now", file=sys.stderr, flush=True)
    data = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", device=dev)
    sd.wait()
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data.tobytes())
    print(f"Saved {out}", file=sys.stderr)
    return out


def read_wav(path: Path) -> tuple[dict, array]:
    with wave.open(str(path), "rb") as w:
        params = {"sample_rate": w.getframerate(), "channels": w.getnchannels(),
                  "sample_width_bytes": w.getsampwidth(), "frames": w.getnframes()}
        raw = w.readframes(w.getnframes())
    samples = array("h")
    if params["sample_width_bytes"] == 2:
        samples.frombytes(raw[: len(raw) - len(raw) % 2])
    params["duration_s"] = round(params["frames"] / params["sample_rate"], 3)
    return params, samples


def levels(samples: array) -> dict:
    """RMS/peak in dBFS plus a crude 'did we capture speech?' verdict."""
    if not samples:
        return {"rms_dbfs": None, "peak_dbfs": None, "verdict": "empty recording"}
    peak = max(abs(s) for s in samples)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))

    def db(x: float) -> float:
        return round(20 * math.log10(x / 32768), 1) if x > 0 else -120.0

    clipped = sum(1 for s in samples if abs(s) >= 32767) / len(samples)
    if peak == 0:
        verdict = "digital silence: wrong device, muted, or no permission"
    elif db(peak) < -40:
        verdict = "very quiet: probably no speech (check device/mute/input volume)"
    elif db(rms) < -50:
        verdict = "weak signal: speech may be too quiet for good STT"
    elif clipped > 0.001:
        verdict = "clipping: lower the input volume"
    else:
        verdict = "OK: signal level consistent with speech"
    return {"rms_dbfs": db(rms), "peak_dbfs": db(peak), "clipped_ratio": round(clipped, 5),
            "verdict": verdict}


def mic_check(seconds: float, device, out: Path) -> dict:
    record(seconds, out, device)
    params, samples = read_wav(out)
    ok_format = (params["sample_rate"], params["channels"], params["sample_width_bytes"]) == (
        SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH)
    return {"file": str(out), "exists": out.exists(), "size_bytes": out.stat().st_size,
            **params, "format_ok_16k_mono_16bit": ok_format, **levels(samples)}


# ------------------------------------------------------------------- transcription ----
def transcribe(wav: Path, model_size: str, threads: int, beam_size: int, language: str,
               local_only: bool) -> dict:
    import numpy as np
    from faster_whisper import WhisperModel

    report: dict = {"model": model_size, "device": "cpu", "compute_type": "int8",
                    "cpu_threads": threads, "num_workers": 1, "logical_cpus": os.cpu_count(),
                    "beam_size": beam_size, "forced_language": language, "file": str(wav),
                    "warnings": []}
    params, samples = read_wav(wav)
    report["audio_duration_s"] = params["duration_s"]
    if (params["sample_rate"], params["channels"], params["sample_width_bytes"]) == (
            SAMPLE_RATE, CHANNELS, SAMPLE_WIDTH):
        audio = np.frombuffer(samples.tobytes(), dtype=np.int16).astype(np.float32) / 32768.0
    else:  # let faster-whisper decode/resample other formats itself
        report["warnings"].append(f"non-standard WAV {params}; decoded by faster-whisper")
        audio = str(wav)

    log_records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                log_records.append(f"{record.name}: {record.getMessage()}")

    handler = _Collect()
    logging.getLogger().addHandler(handler)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=threads,
                             num_workers=1, local_files_only=local_only)
        report["load_time_s"] = round(time.perf_counter() - t0, 2)
        t1 = time.perf_counter()
        segments, info = model.transcribe(audio, language=language, beam_size=beam_size,
                                          vad_filter=False, condition_on_previous_text=False)
        seg_list = list(segments)  # decoding happens while iterating
        report["transcription_time_s"] = round(time.perf_counter() - t1, 2)
    logging.getLogger().removeHandler(handler)
    report["warnings"] += [str(w.message) for w in caught] + log_records

    dur = report["audio_duration_s"] or 0
    report["real_time_factor"] = round(report["transcription_time_s"] / dur, 3) if dur else None
    report["language"] = info.language
    report["language_probability"] = round(info.language_probability, 3)
    report["text"] = " ".join(s.text.strip() for s in seg_list).strip()
    report["segments"] = [{"start": round(s.start, 2), "end": round(s.end, 2),
                           "text": s.text.strip(), "avg_logprob": round(s.avg_logprob, 3),
                           "no_speech_prob": round(s.no_speech_prob, 3)} for s in seg_list]
    return report


# ---------------------------------------------------------------------------- CLI -----
def _default_out(prefix: str) -> Path:
    return RECORDINGS / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"


def _print(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    for k, v in report.items():
        if k == "segments":
            for s in v:
                print(f"  [{s['start']:6.2f}-{s['end']:6.2f}] {s['text']}")
        else:
            print(f"{k:24} {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list capture devices")
    mc = sub.add_parser("mic-check", help="record a short clip and check levels")
    mc.add_argument("--seconds", type=float, default=5.0)
    mc.add_argument("--device", help="input device index or name (default: system default)")
    mc.add_argument("--out", type=Path)
    rec = sub.add_parser("record", help="record a clip for benchmarking")
    rec.add_argument("--seconds", type=float, required=True)
    rec.add_argument("--device")
    rec.add_argument("--out", type=Path)
    tr = sub.add_parser("transcribe", help="transcribe a WAV with faster-whisper (CPU, int8)")
    tr.add_argument("wav", type=Path)
    tr.add_argument("--model", choices=MODELS, required=True)
    tr.add_argument("--threads", type=int, default=8)
    tr.add_argument("--beam-size", type=int, default=5)
    tr.add_argument("--language", default="az")
    tr.add_argument("--local-files-only", action="store_true",
                    help="never download; fail if the model is not cached")
    for sp in (mc, tr):
        sp.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        if args.cmd == "devices":
            info = list_devices()
            for d in info["inputs"]:
                mark = "*" if d["is_default"] else " "
                print(f"{mark} [{d['index']}] {d['name']}  ({d['hostapi']}, "
                      f"{d['max_input_channels']} ch, {d['default_samplerate']:.0f} Hz)")
            print(f"default input index: {info['default_input']}")
        elif args.cmd == "mic-check":
            _print(mic_check(args.seconds, args.device, args.out or _default_out("miccheck")),
                   args.json)
        elif args.cmd == "record":
            record(args.seconds, args.out or _default_out("clip"), args.device)
        elif args.cmd == "transcribe":
            if not args.wav.exists():
                p.error(f"no such file: {args.wav}")
            if not 1 <= args.threads <= (os.cpu_count() or 1):
                p.error(f"--threads must be between 1 and {os.cpu_count()}")
            _print(transcribe(args.wav, args.model, args.threads, args.beam_size,
                              args.language, args.local_files_only), args.json)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # report failures (model download, device errors) plainly
        print(json.dumps({"error": type(e).__name__, "message": str(e)[:500]}, indent=2),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
