#!/usr/bin/env python3
"""Wake-word spike for "Hey ELARA" (isolated engineering tool, NOT production code).

Engine: openWakeWord's frozen feature front end (melspectrogram.onnx + Google's
speech-embedding model embedding_model.onnx), run directly with onnxruntime, which
faster-whisper already installs. No openwakeword package, no scipy/scikit-learn, no torch.

No "Hey ELARA" model exists yet, so this spike uses few-shot enrollment: a handful of
recordings of the phrase become embedding templates, and live audio is matched against
them every 80 ms with DTW on cosine distance. Whisper is never involved; audio stays on
this machine. Enrollment stores embeddings only (no audio) unless --save-wav is given.

Subcommands:
  fetch-models                         download the two feature models (~2.4 MB, sha256-pinned)
  enroll  [--count 5] [--neg-seconds 10] record templates (+ optional background) -> .npz
  listen  [--seconds 60]               live microphone detection with latency/CPU stats
  eval    --pos A.wav ... --neg B.wav  offline detection rate / false activations on WAVs
  selftest                             synthetic end-to-end check with espeak-ng voices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK = 1280                      # 80 ms: one embedding per chunk (openWakeWord's hop)
MEL_CONTEXT = 12_960              # samples fed to the mel model per step (>= 76 mel frames)
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"       # git-ignored
DEFAULT_TEMPLATES = HERE / "recordings" / "hey_elara_templates.npz"
MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"
FEATURE_MODELS = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
GATE_RMS = 200.0                  # AC RMS; same level as production VAD_MIN_RMS
REFRACTORY_S = 1.5
WARP = (0.8, 1.0, 1.25)           # live window lengths relative to a template


# ------------------------------------------------------------------------- models ----
def fetch_models(model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, sha in FEATURE_MODELS.items():
        path = model_dir / name
        if not path.exists():
            print(f"downloading {name} ...")
            url = MODEL_URL.format(name)  # fixed https URL; content is sha256-verified below
            with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
                path.write_bytes(r.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != sha:
            path.unlink()
            sys.exit(f"{name}: sha256 mismatch ({digest}); file removed")
        print(f"ok  {path}  ({path.stat().st_size / 1e6:.1f} MB)")


class Embedder:
    """Streaming openWakeWord features: one 96-dim embedding per 80 ms chunk."""

    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 1):
        import onnxruntime as ort
        missing = [n for n in FEATURE_MODELS if not (model_dir / n).exists()]
        if missing:
            sys.exit(f"missing {missing}; run: python bench/voice/wakeword_bench.py fetch-models")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        cpu = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(str(model_dir / "melspectrogram.onnx"), opts, providers=cpu)
        self.emb = ort.InferenceSession(str(model_dir / "embedding_model.onnx"), opts,
                                        providers=cpu)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(MEL_CONTEXT, np.float32)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        x = chunk.astype(np.float32)
        x -= x.mean()                          # drop the mic's DC offset, as production does
        self._buf = np.concatenate([self._buf[len(chunk):], x])
        mel = self.mel.run(None, {"input": self._buf[None]})[0].squeeze()[-76:]
        mel = mel / 10.0 + 2.0                                   # openWakeWord's scaling
        e = self.emb.run(None, {"input_1": mel[None, :, :, None]})[0].reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    def clip(self, pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Embeddings and AC RMS per 80 ms chunk for a whole clip."""
        self.reset()
        chunks = _chunks(pcm)
        return (np.stack([self.push(c) for c in chunks]),
                np.array([ac_rms(c) for c in chunks]))


def _chunks(pcm: np.ndarray) -> np.ndarray:
    """80 ms chunks; the tail is padded with the last sample (no fake DC step)."""
    pad = (-len(pcm)) % CHUNK
    return np.pad(pcm, (0, pad), mode="edge").reshape(-1, CHUNK)


def ac_rms(chunk: np.ndarray) -> float:
    return float(np.std(chunk.astype(np.float64)))  # DC-free, like production frame_levels


# ------------------------------------------------------------------------ matching ----
def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Length-normalised DTW distance between two sequences of unit vectors."""
    cost = 1.0 - a @ b.T
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = cost[i - 1, j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / (n + m))


def phrase_template(emb: np.ndarray, rms: np.ndarray, gate: float = GATE_RMS) -> np.ndarray | None:
    """The embeddings covering the spoken phrase in an enrollment clip."""
    loud = np.flatnonzero(rms > max(gate, 0.15 * rms.max()))
    if len(loud) == 0:
        return None
    start, end = loud[0], min(len(emb), loud[-1] + 4)  # window still holds the phrase after it
    return emb[start:end] if end - start >= 4 else None


def score(history: np.ndarray, templates: list[np.ndarray]) -> float:
    """Best (lowest) distance of the most recent embeddings to any template."""
    best = np.inf
    for t in templates:
        for w in WARP:
            n = max(3, round(len(t) * w))
            if len(history) >= n:
                best = min(best, dtw(history[-n:], t))
    return best


class Detector:
    def __init__(self, templates: list[np.ndarray], threshold: float, gate: float = GATE_RMS):
        self.templates, self.threshold, self.gate = templates, threshold, gate
        self.keep = max(round(len(t) * WARP[-1]) for t in templates)
        self.hist: list[np.ndarray] = []
        self.rms: list[float] = []
        self.cooldown = 0

    def step(self, emb: np.ndarray, rms: float) -> float | None:
        """Feed one chunk; returns the distance when the wake word fires."""
        self.hist = (self.hist + [emb])[-self.keep:]
        self.rms = (self.rms + [rms])[-self.keep:]
        if self.cooldown:
            self.cooldown -= 1
            return None
        if max(self.rms) < self.gate:          # nothing loud in the window: skip DTW
            return None
        s = score(np.stack(self.hist), self.templates)
        if s <= self.threshold:
            self.cooldown = round(REFRACTORY_S * SAMPLE_RATE / CHUNK)
            self.hist, self.rms = [], []
            return s
        return None


def best_distance(emb: np.ndarray, templates: list[np.ndarray]) -> float:
    keep = max(round(len(t) * WARP[-1]) for t in templates)
    return min((score(emb[max(0, i - keep):i], templates) for i in range(1, len(emb) + 1)),
               default=np.inf)


def calibrate(clips: list[tuple[np.ndarray, np.ndarray]], negatives: list[np.ndarray],
              margin: float) -> dict:
    """Leave-one-out positive distances (and background distances) -> threshold."""
    templates = [phrase_template(e, r) for e, r in clips]
    ok = [(t, e) for t, (e, _) in zip(templates, clips, strict=True) if t is not None]
    if len(ok) < 2:
        sys.exit("need at least 2 enrollment clips with audible speech")
    pos = [best_distance(e, [t for j, (t, _) in enumerate(ok) if j != i])
           for i, (_, e) in enumerate(ok)]
    tmpl = [t for t, _ in ok]
    neg = [best_distance(e, tmpl) for e in negatives]
    if neg and min(neg) > max(pos):
        threshold = (max(pos) + min(neg)) / 2
    else:
        threshold = max(pos) * margin
    return {"templates": tmpl, "threshold": float(threshold), "positives": pos, "negatives": neg}


# --------------------------------------------------------------------------- audio ----
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    if rate != SAMPLE_RATE:  # linear resampling is adequate for a spike
        t = np.arange(0, len(x) / rate, 1 / SAMPLE_RATE)
        x = np.interp(t, np.arange(len(x)) / rate, x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.astype(np.int16).tobytes())


def _sd():
    try:
        import sounddevice as sd
    except (OSError, ImportError) as e:
        sys.exit(f"sounddevice/PortAudio unavailable ({e}); sudo apt install libportaudio2")
    return sd


def record(seconds: float, device) -> np.ndarray:
    sd = _sd()
    x = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16",
               device=device)
    sd.wait()
    return x.reshape(-1)


def _device(d):
    return int(d) if isinstance(d, str) and d.isdigit() else d


# ------------------------------------------------------------------------ commands ----
def save_templates(path: Path, cal: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, threshold=cal["threshold"], positives=np.array(cal["positives"]),
             negatives=np.array(cal["negatives"]),
             **{f"tmpl_{i:02d}": t for i, t in enumerate(cal["templates"])})


def load_templates(path: Path) -> tuple[list[np.ndarray], float]:
    if not path.exists():
        sys.exit(f"{path} not found; run the enroll command first")
    z = np.load(path)
    return [z[k] for k in sorted(z.files) if k.startswith("tmpl_")], \
        float(z["threshold"])


def report_calibration(cal: dict) -> None:
    print(f"templates: {len(cal['templates'])}  "
          f"lengths: {[len(t) for t in cal['templates']]} x 80 ms")
    print(f"positive (leave-one-out) distances: {np.round(cal['positives'], 3).tolist()}")
    if cal["negatives"]:
        print(f"background min distance: {np.round(min(cal['negatives']), 3)}")
    print(f"threshold: {cal['threshold']:.3f}")


def cmd_enroll(a) -> int:
    emb = Embedder(threads=a.threads)
    clips, negatives = [], []
    if a.from_wav:
        pcms = [read_wav(Path(p)) for p in a.from_wav]
    else:
        pcms = []
        for i in range(a.count):
            input(f"[{i + 1}/{a.count}] press Enter, then say 'Hey ELARA' once ... ")
            print(f"  recording {a.seconds:.1f} s")
            pcms.append(record(a.seconds, _device(a.device)))
            if a.save_wav:
                write_wav(Path(a.save_wav) / f"hey_elara_{i + 1}.wav", pcms[-1])
        if a.neg_seconds:
            input(f"press Enter, then talk normally WITHOUT saying the wake word for "
                  f"{a.neg_seconds:.0f} s (or stay quiet) ... ")
            neg = record(a.neg_seconds, _device(a.device))
            negatives.append(emb.clip(neg)[0])
    clips = [emb.clip(p) for p in pcms]
    for p in a.neg_wav or []:
        negatives.append(emb.clip(read_wav(Path(p)))[0])
    cal = calibrate(clips, negatives, a.margin)
    save_templates(Path(a.out), cal)
    report_calibration(cal)
    print(f"saved {a.out} (embeddings only, no audio)")
    return 0


def run_stream(pcm: np.ndarray, det: Detector, emb: Embedder) -> list[dict]:
    """Offline streaming over a clip; latency = detection time - end of the loud region."""
    emb.reset()
    hits, last_loud = [], None
    for i, c in enumerate(_chunks(pcm)):
        r = ac_rms(c)
        if r > det.gate:
            last_loud = i
        s = det.step(emb.push(c), r)
        if s is not None:
            t = (i + 1) * CHUNK / SAMPLE_RATE
            lat = None if last_loud is None else (i - last_loud) * CHUNK / SAMPLE_RATE * 1000
            hits.append({"t_s": round(t, 2), "distance": round(s, 3), "after_speech_ms": lat})
    return hits


def cmd_eval(a) -> int:
    templates, threshold = load_templates(Path(a.templates))
    threshold = a.threshold or threshold
    emb = Embedder(threads=a.threads)
    out = {"threshold": threshold, "positives": [], "negatives": []}
    audio_s = 0.0
    t0 = time.process_time()
    for kind, files in (("positives", a.pos or []), ("negatives", a.neg or [])):
        for f in files:
            pcm = read_wav(Path(f))
            audio_s += len(pcm) / SAMPLE_RATE
            hits = run_stream(pcm, Detector(templates, threshold), emb)
            out[kind].append({"file": f, "seconds": round(len(pcm) / SAMPLE_RATE, 1),
                              "detections": hits})
    cpu = time.process_time() - t0
    out["cpu_percent_of_one_core"] = round(100 * cpu / max(audio_s, 1e-9), 1)
    _print_eval(out, a.json)
    return 0


def _print_eval(out: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(out, indent=2))
        return
    pos, neg = out["positives"], out["negatives"]
    if pos:
        found = sum(bool(p["detections"]) for p in pos)
        lats = [p["detections"][0]["after_speech_ms"] for p in pos if p["detections"]]
        print(f"detected: {found}/{len(pos)} positive clips"
              + (f"; latency after speech end: median {np.median(lats):.0f} ms, "
                 f"max {max(lats):.0f} ms" if lats else ""))
        for p in pos:
            print(f"  {'HIT ' if p['detections'] else 'MISS'} {p['file']} {p['detections'][:1]}")
    if neg:
        n = sum(len(x["detections"]) for x in neg)
        minutes = sum(x["seconds"] for x in neg) / 60
        print(f"false activations: {n} in {minutes:.1f} min of negatives")
        for x in neg:
            if x["detections"]:
                print(f"  FALSE {x['file']} {x['detections']}")
    print(f"threshold {out['threshold']:.3f}; CPU {out['cpu_percent_of_one_core']}% of one "
          f"core (offline, per second of audio)")


def cmd_listen(a) -> int:
    templates, threshold = load_templates(Path(a.templates))
    det = Detector(templates, a.threshold or threshold)
    emb = Embedder(threads=a.threads)
    sd = _sd()
    q: queue.Queue = queue.Queue()
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                            device=_device(a.device),
                            callback=lambda indata, *_: q.put(indata[:, 0].copy()))
    print(f"listening {a.seconds:.0f} s for 'Hey ELARA' (threshold {det.threshold:.3f}); "
          "Ctrl+C to stop")
    steps, step_ms, hits = 0, [], 0
    last_loud_t = None
    wall0, cpu0 = time.monotonic(), time.process_time()
    try:
        with stream:
            while time.monotonic() - wall0 < a.seconds:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                t_chunk = time.monotonic()
                r = ac_rms(chunk)
                if r > det.gate:
                    last_loud_t = t_chunk
                s = det.step(emb.push(chunk), r)
                done = time.monotonic()
                step_ms.append((done - t_chunk) * 1000)
                steps += 1
                if s is not None:
                    hits += 1
                    lat = (done - last_loud_t) * 1000 if last_loud_t else float("nan")
                    print(f"  WAKE  t={done - wall0:6.2f}s  distance={s:.3f}  "
                          f"{lat:.0f} ms after last loud chunk  (compute {step_ms[-1]:.1f} ms)",
                          flush=True)
    except KeyboardInterrupt:
        pass
    wall, cpu = time.monotonic() - wall0, time.process_time() - cpu0
    print(f"\n{steps} chunks in {wall:.1f} s, detections: {hits}")
    if step_ms:
        print(f"compute per 80 ms chunk: median {np.median(step_ms):.1f} ms, "
              f"p95 {np.percentile(step_ms, 95):.1f} ms, max {max(step_ms):.1f} ms")
    print(f"process CPU: {100 * cpu / max(wall, 1e-9):.1f}% of one core")
    return 0


# ----------------------------------------------------------------------- selftest ----
def _espeak(text: str, voice: str, speed: int, pitch: int, tmp: Path) -> np.ndarray:
    f = tmp / "s.wav"
    exe = shutil.which("espeak-ng") or "espeak-ng"
    subprocess.run([exe, "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", str(f),
                    text], check=True, capture_output=True)
    return read_wav(f)


def _scene(speech: np.ndarray, rng: np.random.Generator, noise: float = 40,
           dc: int = 7800) -> np.ndarray:
    """1 s room noise + speech + 1.5 s room noise, with the laptop's DC offset."""
    pad = lambda n: rng.normal(0, noise, n)  # noqa: E731
    x = np.concatenate([pad(SAMPLE_RATE), speech.astype(np.float64) + pad(len(speech)),
                        pad(SAMPLE_RATE * 3 // 2)]) + dc
    return np.clip(x, -32768, 32767).astype(np.int16)


def cmd_selftest(a) -> int:
    if not shutil.which("espeak-ng"):
        sys.exit("selftest needs espeak-ng (sudo apt install espeak-ng)")
    rng = np.random.default_rng(7)
    phrase = "Hey Elara"
    enroll = [("en-us", 160, 50), ("en-us", 140, 40), ("en-gb", 160, 50), ("en-us", 180, 60),
              ("en-gb", 145, 45)]
    positives = [("en-us", 150, 55), ("en-gb", 170, 40), ("en-gb-x-rp", 160, 50),
                 ("en-us", 130, 50), ("en-gb-scotland", 160, 50), ("en-029", 160, 50)]
    negatives = ["Hey Sarah", "Hello there", "Hey Alexa", "Clara", "Hey Siri",
                 "Elevator", "A lot of rain", "Hey, how are you",
                 "What time is it", "Turn off the light in the kitchen",
                 "The weather in Baku is nice today", "Let's go to the library later"]
    emb = Embedder(threads=a.threads)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        clips = [emb.clip(_scene(_espeak(phrase, v, s, p, tmp), rng)) for v, s, p in enroll]
        cal = calibrate(clips, [], a.margin)
        report_calibration(cal)
        out = {"threshold": cal["threshold"], "positives": [], "negatives": []}
        t0, audio_s = time.process_time(), 0.0
        for v, s, p in positives:
            pcm = _scene(_espeak(phrase, v, s, p, tmp), rng)
            audio_s += len(pcm) / SAMPLE_RATE
            out["positives"].append({"file": f"{phrase!r} {v} s{s} p{p}", "seconds": len(pcm) / 16e3,
                                     "detections": run_stream(pcm, Detector(cal["templates"],
                                                                            cal["threshold"]), emb)})
        for text in negatives:
            for v in ("en-us", "en-gb"):
                pcm = _scene(_espeak(text, v, 160, 50, tmp), rng)
                audio_s += len(pcm) / SAMPLE_RATE
                out["negatives"].append({"file": f"{text!r} {v}", "seconds": len(pcm) / 16e3,
                                         "detections": run_stream(
                                             pcm, Detector(cal["templates"], cal["threshold"]),
                                             emb)})
        silence = np.clip(rng.normal(0, 40, 60 * SAMPLE_RATE) + 7800, -32768, 32767)
        audio_s += 60
        out["negatives"].append({"file": "60 s room noise + DC", "seconds": 60.0,
                                 "detections": run_stream(silence.astype(np.int16),
                                                          Detector(cal["templates"],
                                                                   cal["threshold"]), emb)})
        out["cpu_percent_of_one_core"] = round(100 * (time.process_time() - t0) / audio_s, 1)
    _print_eval(out, a.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-models")

    def common(sp, templates=True):
        sp.add_argument("--threads", type=int, default=1, help="onnxruntime threads (default 1)")
        sp.add_argument("--json", action="store_true")
        if templates:
            sp.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
            sp.add_argument("--threshold", type=float, help="override the calibrated threshold")

    e = sub.add_parser("enroll")
    common(e, templates=False)
    e.add_argument("--count", type=int, default=5)
    e.add_argument("--seconds", type=float, default=2.0)
    e.add_argument("--neg-seconds", type=float, default=0.0,
                   help="also record this much background speech for calibration")
    e.add_argument("--device")
    e.add_argument("--from-wav", nargs="+", help="enroll from WAV files instead of the mic")
    e.add_argument("--neg-wav", nargs="+", help="background WAVs for calibration")
    e.add_argument("--save-wav", help="directory to keep enrollment audio (off by default)")
    e.add_argument("--margin", type=float, default=1.15)
    e.add_argument("--out", default=str(DEFAULT_TEMPLATES))
    ls = sub.add_parser("listen")
    common(ls)
    ls.add_argument("--seconds", type=float, default=60.0)
    ls.add_argument("--device")
    ev = sub.add_parser("eval")
    common(ev)
    ev.add_argument("--pos", nargs="+")
    ev.add_argument("--neg", nargs="+")
    st = sub.add_parser("selftest")
    common(st, templates=False)
    st.add_argument("--margin", type=float, default=1.15)
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    a = build_parser().parse_args(argv)
    if a.cmd == "fetch-models":
        fetch_models()
        return 0
    return {"enroll": cmd_enroll, "listen": cmd_listen, "eval": cmd_eval,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
