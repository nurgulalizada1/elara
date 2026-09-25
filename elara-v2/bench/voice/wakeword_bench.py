#!/usr/bin/env python3
"""Wake-word spike for "Hey ELARA" (isolated engineering tool, NOT production code).

Engine: openWakeWord's frozen feature front end (melspectrogram.onnx + Google's
speech-embedding model embedding_model.onnx), run directly with onnxruntime, which
faster-whisper already installs. No openwakeword package, no scipy/scikit-learn, no torch.

No "Hey ELARA" model exists yet, so this spike uses few-shot enrollment: recordings of the
phrase become embedding templates, and live audio is matched against them every 80 ms with
subsequence DTW on cosine distance. Whisper is never involved; audio stays on this machine.

Subcommands:
  fetch-models                 download the two feature models (~2.4 MB, sha256-pinned)
  collect  --out DIR           guided, labelled recording session on the real mic
  synth    --out DIR           synthetic labelled dataset with espeak-ng voices
  evaluate --set DIR           enroll + calibrate + score every decision rule: TP/FP/FN,
                               precision, recall, false activations/hour, latency, CPU
  listen   --templates F.npz   live detection with per-candidate diagnostics
  scan     --templates F.npz WAV...   candidates (with diagnostics) in recorded WAVs

A dataset directory holds WAVs plus manifest.json:
  {"items": [{"file": "...wav", "role": "enroll|calib_negative|positive|negative",
              "tag": "normal|fast|hey alexa|tv|...", "speaker": "..."}]}
Each "positive" clip contains exactly one "Hey ELARA"; negatives contain none.
"""
# ruff: noqa: E402  (thread caps must be set before numpy/onnxruntime are imported)

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # BLAS thread pools start at import time

import argparse
import hashlib
import json
import math
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK = 1280                      # 80 ms: one embedding per chunk (openWakeWord's hop)
CHUNK_S = CHUNK / SAMPLE_RATE
MEL_NEW = CHUNK + 480             # mel input per step -> exactly 8 new 10 ms mel frames
MEL_FRAMES = 76                   # embedding model input: 76 mel frames (~0.76 s)
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"       # git-ignored
DEFAULT_TEMPLATES = HERE / "recordings" / "hey_elara_templates.npz"
MODEL_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/{}"
FEATURE_MODELS = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
GATE_RMS = 200.0                  # AC RMS; same level as production VAD_MIN_RMS
REFRACTORY = round(1.5 / CHUNK_S)  # chunks after a detection in which nothing fires
REPEAT_S = 3.0                    # live: a candidate this close to the previous = same event
MAX_STRETCH = 1.7                 # history searched per template, relative to its length
LEGACY_WARP = (0.8, 1.0, 1.25)    # original spike: three fixed window lengths
TRUTH_TAIL = round(1.0 / CHUNK_S)  # a detection up to 1 s after the phrase belongs to it


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
    """Streaming openWakeWord features: one unit-norm 96-dim embedding per 80 ms chunk.

    The mel model only sees the new chunk plus 480 samples of context (8 new frames), and
    the 76-frame mel window is a reused buffer; this is exactly what recomputing the mel
    spectrogram over the whole 0.8 s window gives, at ~1/7 of the mel cost.
    """

    def __init__(self, model_dir: Path = MODEL_DIR, threads: int = 1):
        import onnxruntime as ort
        missing = [n for n in FEATURE_MODELS if not (model_dir / n).exists()]
        if missing:
            sys.exit(f"missing {missing}; run: python bench/voice/wakeword_bench.py fetch-models")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        cpu = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(str(model_dir / "melspectrogram.onnx"), opts, providers=cpu)
        self.emb = ort.InferenceSession(str(model_dir / "embedding_model.onnx"), opts,
                                        providers=cpu)
        self._silence = self._mel(np.zeros(MEL_NEW, np.float32))[0]
        self.reset()

    def _mel(self, x: np.ndarray) -> np.ndarray:
        return self.mel.run(None, {"input": x[None]})[0].reshape(-1, 32) / 10.0 + 2.0

    def reset(self) -> None:
        self._raw = np.zeros(MEL_NEW, np.float32)
        self._mels = np.tile(self._silence, (MEL_FRAMES, 1)).astype(np.float32)
        self._input = self._mels.reshape(1, MEL_FRAMES, 32, 1)  # view, no copy per step

    def push(self, chunk: np.ndarray) -> np.ndarray:
        raw = self._raw
        raw[:-CHUNK] = raw[CHUNK:]                  # keep 480 samples of context
        raw[-CHUNK:] = chunk
        raw[-CHUNK:] -= raw[-CHUNK:].mean()         # drop the mic's DC offset, as production
        new = self._mel(raw)
        self._mels[:-len(new)] = self._mels[len(new):]
        self._mels[-len(new):] = new
        e = self.emb.run(None, {"input_1": self._input})[0].reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    def clip(self, pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Embeddings, AC RMS per chunk and feature compute seconds for a whole clip."""
        self.reset()
        chunks = _chunks(pcm)
        t0 = time.perf_counter()
        emb = np.stack([self.push(c) for c in chunks])
        return emb, np.array([ac_rms(c) for c in chunks]), time.perf_counter() - t0


def _chunks(pcm: np.ndarray) -> np.ndarray:
    """80 ms chunks; the tail is padded with the last sample (no fake DC step)."""
    pad = (-len(pcm)) % CHUNK
    return np.pad(pcm, (0, pad), mode="edge").reshape(-1, CHUNK)


def ac_rms(chunk: np.ndarray) -> float:
    return float(np.std(chunk.astype(np.float64)))  # DC-free, like production frame_levels


# ------------------------------------------------------------------------ matching ----
def dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Original spike: length-normalised DTW, both ends fixed (kept for comparison)."""
    cost = 1.0 - a @ b.T
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = cost[i - 1, j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / (n + m))


def legacy_score(history: np.ndarray, templates: list[np.ndarray]) -> float:
    """Original spike: min over templates x 3 fixed window lengths."""
    best = np.inf
    for t in templates:
        for w in LEGACY_WARP:
            n = max(3, round(len(t) * w))
            if len(history) >= n:
                best = min(best, dtw(history[-n:], t))
    return best


def subseq_dtw(cost: list[list[float]]) -> tuple[float, int]:
    """Template (rows) matched completely, ending at the newest history frame (last column)
    and starting at any history frame. Returns (mean cost along the path, start column).

    One pass covers every window length, replacing the three fixed warps; plain Python
    lists because per-element numpy indexing is several times slower at this size.
    """
    n = len(cost[0])
    d, length, start = list(cost[0]), [1] * n, list(range(n))
    for row in cost[1:]:
        nd, nl, ns = [0.0] * n, [0] * n, [0] * n
        nd[0], nl[0], ns[0] = row[0] + d[0], length[0] + 1, start[0]
        for j in range(1, n):
            diag, up, left = d[j - 1], d[j], nd[j - 1]
            if diag <= up and diag <= left:
                nd[j], nl[j], ns[j] = row[j] + diag, length[j - 1] + 1, start[j - 1]
            elif up <= left:
                nd[j], nl[j], ns[j] = row[j] + up, length[j] + 1, start[j]
            else:
                nd[j], nl[j], ns[j] = row[j] + left, nl[j - 1] + 1, ns[j - 1]
        d, length, start = nd, nl, ns
    return d[-1] / length[-1], start[-1]


class Matcher:
    """Distance of the phrase ending *now* to every template, plus the matched duration."""

    def __init__(self, templates: list[np.ndarray]):
        self.templates = templates
        self.lens = np.array([len(t) for t in templates])
        self.stack = np.concatenate(templates)
        self.bounds = np.cumsum([0, *self.lens])
        self.windows = [round(m * MAX_STRETCH) for m in self.lens]
        self.keep = max(self.windows)

    def __call__(self, history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cost = 1.0 - self.stack @ history.T         # all templates in one product
        dist, span = [], []
        for t in range(len(self.templates)):
            w = min(self.windows[t], history.shape[0])
            c = cost[self.bounds[t]:self.bounds[t + 1], -w:].tolist()
            d, s = subseq_dtw(c)
            dist.append(d)
            span.append(w - s)                      # history chunks the phrase occupied
        return np.array(dist), np.array(span)


@dataclass(frozen=True)
class Rule:
    """How per-chunk template distances become a wake decision."""
    name: str
    k: int = 1                                    # score = mean of the k nearest templates
    min_run: int = 1                              # consecutive qualifying chunks to fire
    span: tuple[float, float] | None = None       # matched duration / template duration
    legacy: bool = False                          # original scorer (for comparison)


RULES = {
    "legacy": Rule("legacy", legacy=True),        # what ran on the laptop (e3d01d4)
    "nearest": Rule("nearest"),                   # new DTW, single nearest template/frame
    "k2": Rule("k2", k=2),                        # mean of the 2 nearest templates
    "robust": Rule("robust", k=2, min_run=2, span=(0.6, 1.6)),
}


@dataclass
class Trace:
    """Per-chunk matching results for one clip (threshold-independent, so cached)."""
    rms: np.ndarray
    gate: np.ndarray            # window had audio above the energy gate
    legacy: np.ndarray          # legacy score (inf when not computed)
    dist: np.ndarray            # (chunks, templates) subsequence-DTW distances
    span: np.ndarray            # (chunks, templates) matched duration in chunks
    feature_s: float
    match_s: float
    legacy_s: float


def trace_clip(emb: np.ndarray, rms: np.ndarray, matcher: Matcher, feature_s: float = 0.0,
               with_legacy: bool = True) -> Trace:
    n, t = len(emb), len(matcher.templates)
    gate = np.zeros(n, bool)
    legacy = np.full(n, np.inf)
    dist, span = np.full((n, t), np.inf), np.zeros((n, t))
    legacy_keep = max(round(m * LEGACY_WARP[-1]) for m in matcher.lens)
    match_s = legacy_s = 0.0
    for i in range(n):
        lo = max(0, i + 1 - matcher.keep)
        if rms[lo:i + 1].max() < GATE_RMS:        # nothing loud in the window: no matching
            continue
        gate[i] = True
        t0 = time.perf_counter()
        dist[i], span[i] = matcher(emb[lo:i + 1])
        t1 = time.perf_counter()
        match_s += t1 - t0
        if with_legacy:
            legacy[i] = legacy_score(emb[max(0, i + 1 - legacy_keep):i + 1], matcher.templates)
            legacy_s += time.perf_counter() - t1
    return Trace(rms, gate, legacy, dist, span, feature_s, match_s, legacy_s)


def rule_scores(tr: Trace, rule: Rule, lens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-chunk score under a rule (inf = cannot fire) and the best template index."""
    if rule.legacy:
        return tr.legacy.copy(), np.zeros(len(tr.legacy), int)
    order = np.argsort(tr.dist, axis=1)
    k = min(rule.k, tr.dist.shape[1])
    score = np.take_along_axis(tr.dist, order[:, :k], axis=1).mean(axis=1)
    best = order[:, 0]
    if rule.span:
        ratio = tr.span[np.arange(len(best)), best] / lens[best]
        score[(ratio < rule.span[0]) | (ratio > rule.span[1])] = np.inf
    return score, best


def decide(score: np.ndarray, rule: Rule, threshold: float) -> list[int]:
    """Chunks at which the detector fires (with run requirement and refractory period)."""
    fires, run, cooldown = [], 0, 0
    for i, s in enumerate(score):
        if cooldown:
            cooldown -= 1
            run = 0
            continue
        run = run + 1 if s <= threshold else 0
        if run >= rule.min_run:
            fires.append(i)
            run, cooldown = 0, REFRACTORY
    return fires


def trigger_score(score: np.ndarray, rule: Rule, lo: int = 0, hi: int | None = None) -> float:
    """Lowest threshold at which the rule would fire within chunks [lo, hi)."""
    s = score[lo:hi]
    r = rule.min_run
    if len(s) < r:
        return np.inf
    return float(np.min(np.max(np.lib.stride_tricks.sliding_window_view(s, r), axis=1)))


def truth_window(rms: np.ndarray) -> tuple[int, int, int]:
    """(first loud chunk, last loud chunk, end of the acceptance window) of a positive clip."""
    loud = np.flatnonzero(rms > max(GATE_RMS, 0.15 * rms.max()))
    if len(loud) == 0:
        return 0, 0, 0
    return int(loud[0]), int(loud[-1]), int(loud[-1]) + TRUTH_TAIL


def phrase_template(emb: np.ndarray, rms: np.ndarray) -> np.ndarray | None:
    """The embeddings covering the spoken phrase in an enrollment clip."""
    first, last, _ = truth_window(rms)
    if last == first == 0:
        return None
    end = min(len(emb), last + 4)                 # the 0.76 s window still holds the phrase
    return emb[first:end] if end - first >= 4 else None


# -------------------------------------------------------------------- calibration ----
def calibrate(enroll: list[tuple[np.ndarray, np.ndarray]],
              calib_neg: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Templates + a per-rule threshold from enrollment data only (never from test data).

    Positive evidence: leave-one-out trigger scores of the enrollment clips.
    Negative evidence: trigger scores of the calibration background (not a single minimum
    distance, but the level at which each rule would actually fire on it).
    """
    tmpl = [phrase_template(e, r) for e, r in enroll]
    keep = [i for i, t in enumerate(tmpl) if t is not None]
    if len(keep) < 3:
        sys.exit("need at least 3 enrollment clips with audible speech")
    templates = [tmpl[i] for i in keep]
    lens = np.array([len(t) for t in templates])
    matcher = Matcher(templates)
    loo = []
    for j, i in enumerate(keep):
        others = [t for jj, t in enumerate(templates) if jj != j]
        e, r = enroll[i]
        loo.append((trace_clip(e, r, Matcher(others)), np.array([len(t) for t in others]),
                    truth_window(r)))
    neg = [trace_clip(e, r, matcher) for e, r in calib_neg]
    out = {"templates": templates, "rules": {}}
    for name, rule in RULES.items():
        pos = [trigger_score(rule_scores(tr, rule, ln)[0], rule, a, c + 1)
               for tr, ln, (a, _, c) in loo]
        negs = [trigger_score(rule_scores(tr, rule, lens)[0], rule) for tr in neg]
        p_max, n_min = max(pos), min(negs, default=np.inf)
        if p_max < n_min:
            thr = (p_max + n_min) / 2 if np.isfinite(n_min) else p_max * 1.15
            status = "separable" if np.isfinite(n_min) else "no background recorded"
        else:
            thr, status = n_min * 0.98, "OVERLAP: background fires below some positives"
        out["rules"][name] = {"threshold": float(thr), "loo_positive": pos,
                              "background_min": float(n_min), "status": status}
    return out


def save_templates(path: Path, cal: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = {k: v["threshold"] for k, v in cal["rules"].items()}
    np.savez(path, thresholds=json.dumps(thresholds),
             **{f"tmpl_{i:02d}": t for i, t in enumerate(cal["templates"])})


def load_templates(path: Path) -> tuple[list[np.ndarray], dict]:
    if not path.exists():
        sys.exit(f"{path} not found; run `evaluate --save-templates` first")
    z = np.load(path)
    if "thresholds" not in z.files:
        sys.exit(f"{path} is from the old spike format; re-run `evaluate --save-templates`")
    return ([z[k] for k in sorted(z.files) if k.startswith("tmpl_")],
            json.loads(str(z["thresholds"])))


# --------------------------------------------------------------------------- audio ----
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16-bit PCM")
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        rate = w.getframerate()
    if rate != SAMPLE_RATE:  # linear resampling is adequate for a benchmark
        t = np.arange(0, len(x) / rate, 1 / SAMPLE_RATE)
        x = np.interp(t, np.arange(len(x)) / rate, x)
    return np.clip(x, -32768, 32767).astype(np.int16)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(pcm).astype(np.int16).tobytes())


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


# ------------------------------------------------------------------------ datasets ----
HARD_NEGATIVES = ["Hey Alexa", "Hey Sarah", "Hey Laura", "Elara", "What time is it?"]


class Manifest:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "manifest.json"
        self.items = json.loads(self.path.read_text())["items"] if self.path.exists() else []

    def add(self, pcm: np.ndarray, role: str, tag: str, speaker: str = "user") -> None:
        n = sum(1 for it in self.items if it["role"] == role) + 1
        name = f"{speaker}_{role}_{n:03d}_{tag.lower().replace(' ', '-').strip('?')}.wav"
        write_wav(self.root / name, pcm)
        self.items.append({"file": name, "role": role, "tag": tag, "speaker": speaker})
        self.root.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"items": self.items}, indent=1))  # after every clip


def cmd_collect(a) -> int:
    """Guided recording on the real microphone (labels are known because we prompt)."""
    m = Manifest(Path(a.out))
    dev = _device(a.device)

    def take(prompt: str, seconds: float, role: str, tag: str) -> None:
        input(f"{prompt}\n  press Enter, then speak ({seconds:.0f} s) ... ")
        m.add(record(seconds, dev), role, tag)
        print("  saved")

    print(f"Recording a labelled set into {m.root} (local only; delete it when done).")
    for i in range(a.enroll):
        take(f"[enroll {i + 1}/{a.enroll}] say 'Hey ELARA' once, normally", 2.5, "enroll",
             "normal")
    take("[calibration] talk normally WITHOUT the wake word (read something aloud)",
         a.calib_seconds, "calib_negative", "conversation")
    for phrase in (HARD_NEGATIVES if a.calib_hard else []):
        take(f"[calibration] say '{phrase}' once", 2.5, "calib_negative", phrase)
    styles = (["normal"] * 8 + ["fast"] * 3 + ["slow"] * 3 + ["quiet"] * 3 + ["loud"] * 3)
    styles = (styles * (a.positives // len(styles) + 1))[:a.positives]
    for i, style in enumerate(styles):
        take(f"[positive {i + 1}/{len(styles)}] say 'Hey ELARA' once, {style}",
             3.0 if style == "slow" else 2.5, "positive", style)
    for phrase in HARD_NEGATIVES:
        for i in range(a.hard):
            take(f"[hard negative] say '{phrase}' ({i + 1}/{a.hard})", 2.5, "negative", phrase)
    if a.conversation_seconds:
        take("[negative] talk freely, never saying the wake word", a.conversation_seconds,
             "negative", "conversation")
    if a.tv_seconds:
        take("[negative] play TV / a video with speech near the laptop; stay quiet",
             a.tv_seconds, "negative", "tv")
    if a.silence_seconds:
        take("[negative] stay silent (normal room)", a.silence_seconds, "negative", "silence")
    print(f"done: {len(m.items)} clips in {m.path}")
    return 0


SENTENCES = [
    "I think we should leave a bit earlier tomorrow because of the traffic.",
    "The weather in Baku is nice today, but it might rain in the evening.",
    "Could you send me the report before the meeting starts?",
    "My sister is coming to visit next weekend with her kids.",
    "Let's go to the library later and look for that book.",
    "The new restaurant downtown has really good soup.",
    "Turn off the light in the kitchen when you leave.",
    "We need to buy milk, bread, eggs and some fruit.",
    "The football match starts at eight o'clock tonight.",
    "Hello there, how are you doing this morning?",
    "Laura said the elevator in her building is broken again.",
    "Alexandra and Sarah went to the cinema yesterday.",
    "Hey, can you hear me? The connection is really bad.",
    "In today's news, the government announced a new transport plan.",
    "Researchers found that regular exercise improves sleep quality.",
    "The temperature will drop to five degrees overnight.",
    "Clara, a lot of people are waiting outside already.",
    "Please remind me to call the dentist on Monday.",
    "Hey everyone, welcome back to the show.",
    "Our era of fast technology changes how we learn.",
]
SPEAKERS = {  # synthetic "users": each one is enrolled and tested separately
    "us-m": "en-us", "gb-f": "en-gb+f3", "rp-m": "en-gb-x-rp+m3",
}
BACKGROUND_VOICES = ["en-us+f2", "en-gb+m4", "en-029+m3", "en-gb-scotland+f1", "en-us+m7"]


def _espeak(text: str, voice: str, speed: int, pitch: int, tmp: Path) -> np.ndarray:
    f = tmp / "s.wav"
    exe = shutil.which("espeak-ng") or "espeak-ng"
    subprocess.run([exe, "-v", voice, "-s", str(speed), "-p", str(pitch), "-w", str(f), text],
                   check=True, capture_output=True)
    return read_wav(f).astype(np.float64)


def _room(rng: np.random.Generator, seconds: float, noise: float = 40.0) -> np.ndarray:
    return rng.normal(0, noise, int(seconds * SAMPLE_RATE))


def _finish(x: np.ndarray, rng: np.random.Generator, noise: float = 40.0,
            dc: int = 7800) -> np.ndarray:
    """Add room noise and the laptop's DC offset; clip to int16."""
    return np.clip(x + rng.normal(0, noise, len(x)) + dc, -32768, 32767).astype(np.int16)


def _talk(rng, voices, seconds, tmp, gain=1.0) -> np.ndarray:
    parts, total = [], 0.0
    while total < seconds:
        s = _espeak(str(rng.choice(SENTENCES)), str(rng.choice(voices)),
                    int(rng.integers(140, 185)), int(rng.integers(35, 65)), tmp) * gain
        gap = _room(rng, float(rng.uniform(0.2, 1.2)), 0)
        parts += [s, gap]
        total += (len(s) + len(gap)) / SAMPLE_RATE
    return np.concatenate(parts)[:int(seconds * SAMPLE_RATE)]


def cmd_synth(a) -> int:
    if not shutil.which("espeak-ng"):
        sys.exit("synth needs espeak-ng (sudo apt install espeak-ng)")
    rng = np.random.default_rng(a.seed)
    m = Manifest(Path(a.out))
    if m.items:
        sys.exit(f"{m.path} already exists; choose an empty --out")
    pad = lambda x: np.concatenate([_room(rng, 0.8, 0), x, _room(rng, 1.2, 0)])  # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for spk, voice in SPEAKERS.items():
            say = lambda text, speed=160, pitch=50, gain=1.0, v=voice: pad(  # noqa: E731
                _espeak(text, v, speed, pitch, tmp) * gain)
            for sp, pi in [(160, 50), (150, 45), (170, 55), (155, 52), (165, 47)]:
                m.add(_finish(say("Hey Elara", sp, pi), rng), "enroll", "normal", spk)
            m.add(_finish(_talk(rng, [voice], 30, tmp), rng), "calib_negative", "conversation",
                  spk)
            for phrase in HARD_NEGATIVES:  # the user's own takes; test takes are separate
                m.add(_finish(say(phrase, int(rng.integers(150, 176)), int(rng.integers(45, 56))),
                              rng), "calib_negative", phrase, spk)
            for tag, sp, gain in [("slow", 120, 1), ("slow", 135, 1), ("normal", 150, 1),
                                  ("normal", 160, 1), ("normal", 170, 1), ("fast", 190, 1),
                                  ("fast", 210, 1), ("quiet", 160, 0.3), ("quiet", 145, 0.35),
                                  ("loud", 160, 2.2), ("loud", 175, 2.5), ("normal", 158, 0.8)]:
                pitch = int(rng.integers(40, 62))
                m.add(_finish(say("Hey Elara", sp, pitch, gain), rng), "positive", tag, spk)
            for phrase in HARD_NEGATIVES + ["Hey Clara", "Hello Lara", "Hey, Sarah, over here"]:
                for v in (voice, str(rng.choice(BACKGROUND_VOICES))):
                    m.add(_finish(say(phrase, int(rng.integers(145, 180)), 50, 1.0, v), rng),
                          "negative", phrase, spk)
            m.add(_finish(_talk(rng, [voice], a.conversation_seconds, tmp), rng), "negative",
                  "conversation", spk)
            m.add(_finish(_talk(rng, BACKGROUND_VOICES, a.tv_seconds, tmp, gain=0.6), rng,
                          noise=120), "negative", "tv", spk)
            m.add(_finish(_room(rng, a.silence_seconds, 0), rng), "negative", "silence", spk)
            print(f"speaker {spk}: done")
    print(f"{len(m.items)} clips in {m.path}")
    return 0


# ---------------------------------------------------------------------- evaluation ----
def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _classify(fires: list[int], positive: bool, window: tuple[int, int, int]) -> dict:
    """First detection inside a positive's window = TP, later ones there = repeat."""
    first, last, end = window
    out = {"tp": 0, "fp": 0, "repeat": 0, "latency_ms": None}
    for f in fires:
        if positive and first <= f <= end:
            if out["tp"]:
                out["repeat"] += 1
            else:
                out["tp"] = 1
                out["latency_ms"] = (f - last) * CHUNK_S * 1000
        else:
            out["fp"] += 1
    return out


def evaluate_speaker(emb_cache: dict, items: list[dict], speaker: str, neg_items: list[dict],
                     emb: Embedder, legacy: bool) -> dict:
    get = lambda it: emb_cache[it["file"]]  # noqa: E731
    enroll = [get(it)[:2] for it in items if it["role"] == "enroll" and it["speaker"] == speaker]
    calib = [get(it)[:2] for it in items
             if it["role"] == "calib_negative" and it["speaker"] == speaker]
    cal = calibrate(enroll, calib)
    matcher = Matcher(cal["templates"])
    lens = matcher.lens
    tests = [it for it in items if it["role"] == "positive" and it["speaker"] == speaker]
    tests += neg_items
    traces = {}
    for it in tests:
        e, r, fs = get(it)
        traces[it["file"]] = trace_clip(e, r, matcher, fs, with_legacy=legacy)
    res = {"calibration": cal, "rules": {}, "traces": traces}
    for name, rule in RULES.items():
        if rule.legacy and not legacy:
            continue
        thr = cal["rules"][name]["threshold"]
        rows = []
        for it in tests:
            tr = traces[it["file"]]
            score, best = rule_scores(tr, rule, lens)
            pos = it["role"] == "positive"
            window = truth_window(tr.rms) if pos else (0, 0, 0)
            fires = decide(score, rule, thr)
            c = _classify(fires, pos, window)
            c.update(file=it["file"], tag=it["tag"], positive=pos,
                     seconds=len(tr.rms) * CHUNK_S,
                     trigger=trigger_score(score, rule, window[0], window[2] + 1) if pos
                     else trigger_score(score, rule),
                     candidates=[_candidate(score, best, tr, lens, f, thr) for f in fires])
            rows.append(c)
        res["rules"][name] = {"threshold": thr, "rows": rows}
    return res


def _candidate(score, best, tr: Trace, lens, f: int, thr: float) -> dict:
    """Diagnostics for one detection: how isolated or sustained the match was."""
    below = score <= thr
    lo = f
    while lo > 0 and below[lo - 1]:
        lo -= 1
    hi = f
    while hi + 1 < len(below) and below[hi + 1]:
        hi += 1
    near = np.flatnonzero(below[max(0, f - 6):f + 7])
    b = int(best[f])
    return {"t_s": round((f + 1) * CHUNK_S, 2), "distance": round(float(score[f]), 4),
            "threshold": round(thr, 4), "best_template": b,
            "template_distances": np.round(np.sort(tr.dist[f]), 4).tolist()[:3],
            "run_chunks": hi - lo + 1, "nearby_chunks": len(near),
            "duration_s": round(float(tr.span[f, b]) * CHUNK_S, 2),
            "template_s": round(float(lens[b]) * CHUNK_S, 2)}


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["positive"]]
    tp = sum(r["tp"] for r in pos)
    fn = len(pos) - tp
    fp = sum(r["fp"] for r in rows)
    neg_h = sum(r["seconds"] for r in rows if not r["positive"]) / 3600
    lat = [r["latency_ms"] for r in pos if r["latency_ms"] is not None]
    pos_trig = [r["trigger"] for r in pos]
    neg_trig = [r["trigger"] for r in rows if not r["positive"]]
    lo, hi = _wilson(tp, len(pos))
    fa_h = fp / neg_h if neg_h else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "repeats": sum(r["repeat"] for r in rows),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / len(pos) if pos else float("nan"), "recall_ci95": (lo, hi),
            "fa_per_hour": fa_h,
            "fa_per_hour_upper95": (3.0 if fp == 0 else fp + 1.96 * math.sqrt(fp)) / neg_h
            if neg_h else float("nan"),
            "negative_minutes": neg_h * 60,
            # threshold-free: how far the closest negative stays from the positives
            "pos_trigger_p90": float(np.quantile(pos_trig, 0.9)) if pos_trig else None,
            "neg_trigger_min": float(min(neg_trig)) if neg_trig else None,
            "separation": float(min(neg_trig) / np.quantile(pos_trig, 0.9))
            if pos_trig and neg_trig else None,
            "latency_ms_median": float(np.median(lat)) if lat else None,
            "latency_ms_max": float(max(lat)) if lat else None}


def sweep(rows: list[dict], rule: Rule, traces: dict, lens, pos_triggers) -> list[dict]:
    """Recall vs false activations at thresholds placed at positive trigger quantiles
    (insight only: chosen on test data, so NOT a valid operating point)."""
    out = []
    for q in (0.5, 0.8, 0.9, 1.0):
        thr = float(np.quantile(pos_triggers, q))
        fp = 0
        for r in rows:
            if not r["positive"]:
                fp += len(decide(rule_scores(traces[r["file"]], rule, lens)[0], rule, thr))
        out.append({"recall_target": q, "threshold": thr, "fp": fp})
    return out


def cmd_evaluate(a) -> int:
    root = Path(a.set)
    items = Manifest(root).items
    if not items:
        sys.exit(f"no manifest in {root}")
    emb = Embedder(threads=a.threads)
    cache = {}
    for it in items:
        e, r, fs = emb.clip(read_wav(root / it["file"]))
        cache[it["file"]] = (e, r, fs)
    speakers = sorted({it["speaker"] for it in items if it["role"] == "enroll"})
    negatives = [it for it in items if it["role"] == "negative"]
    report = {"speakers": {}, "total": {}, "cpu": {}}
    all_rows: dict[str, list] = {}
    feature_s = sum(v[2] for v in cache.values())
    chunks = sum(len(v[1]) for v in cache.values())
    match_s = legacy_s = 0.0
    gated = traced = 0
    for spk in speakers:
        res = evaluate_speaker(cache, items, spk, negatives, emb, legacy=not a.no_legacy)
        spk_rep = {"calibration": {k: {kk: vv for kk, vv in v.items()}
                                   for k, v in res["calibration"]["rules"].items()},
                   "template_chunks": [len(t) for t in res["calibration"]["templates"]],
                   "rules": {}}
        lens = np.array(spk_rep["template_chunks"])
        for tr in res["traces"].values():
            match_s += tr.match_s
            legacy_s += tr.legacy_s
            gated += int(tr.gate.sum())
            traced += len(tr.gate)
        for name, rr in res["rules"].items():
            s = summarize(rr["rows"])
            pos_trig = [r["trigger"] for r in rr["rows"] if r["positive"]]
            s["sweep"] = sweep(rr["rows"], RULES[name], res["traces"], lens,
                               [p for p in pos_trig if np.isfinite(p)] or [0.0])
            s["threshold"] = rr["threshold"]
            s["false_activations"] = [dict(file=r["file"], tag=r["tag"], **c)
                                      for r in rr["rows"] if r["fp"] for c in r["candidates"]
                                      if not (r["positive"] and r["tp"] and c is r["candidates"][0])]
            s["misses"] = [dict(file=r["file"], tag=r["tag"], trigger=r["trigger"])
                           for r in rr["rows"] if r["positive"] and not r["tp"]]
            spk_rep["rules"][name] = s
            all_rows.setdefault(name, []).extend(rr["rows"])
            if a.save_templates and spk == speakers[0]:
                save_templates(Path(a.save_templates), res["calibration"])
        report["speakers"][spk] = spk_rep
    report["total"] = {name: summarize(rows) for name, rows in all_rows.items()}
    audio_s = chunks * CHUNK_S
    report["cpu"] = {
        "features_ms_per_chunk": 1000 * feature_s / chunks,
        "match_ms_per_gated_chunk": 1000 * match_s / max(gated, 1),
        "legacy_match_ms_per_gated_chunk": 1000 * legacy_s / max(gated, 1) if legacy_s else None,
        "gate_open_fraction": gated / max(traced, 1),
        "audio_minutes": audio_s / 60,
    }
    c = report["cpu"]
    c["cpu_percent_one_core"] = 100 * (c["features_ms_per_chunk"] + c["gate_open_fraction"]
                                       * c["match_ms_per_gated_chunk"]) / (CHUNK_S * 1000)
    if c["legacy_match_ms_per_gated_chunk"] is not None:
        c["legacy_cpu_percent_one_core"] = 100 * (c["features_ms_per_chunk"]
                                                  + c["gate_open_fraction"]
                                                  * c["legacy_match_ms_per_gated_chunk"]) / 80
    if a.json:
        print(json.dumps(report, indent=1, default=_json_default))
    else:
        _print_report(report)
    return 0


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _f(x, fmt="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else fmt.format(x)


def _print_report(rep: dict) -> None:
    for spk, s in rep["speakers"].items():
        print(f"\n=== enrolled speaker: {spk}  (templates {s['template_chunks']} x 80 ms)")
        for name, cal in s["calibration"].items():
            if name not in s["rules"]:
                continue
            r = s["rules"][name]
            print(f"  [{name:7}] thr={_f(cal['threshold'], '{:.4f}')} "
                  f"(LOO pos max {_f(max(cal['loo_positive']), '{:.4f}')}, background min "
                  f"{_f(cal['background_min'], '{:.4f}')}: {cal['status']})")
            print(f"            TP={r['tp']} FN={r['fn']} FP={r['fp']} repeats={r['repeats']} "
                  f"recall={_f(r['recall'], '{:.2f}')} precision={_f(r['precision'], '{:.2f}')} "
                  f"FA/h={_f(r['fa_per_hour'], '{:.1f}')} latency med="
                  f"{_f(r['latency_ms_median'], '{:.0f}')} ms")
            for fa in r["false_activations"][:6]:
                print(f"            FALSE {fa['tag']!r:28} t={fa['t_s']}s d={fa['distance']} "
                      f"run={fa['run_chunks']} nearby={fa['nearby_chunks']} "
                      f"dur={fa['duration_s']}s/{fa['template_s']}s tmpl={fa['best_template']}")
            for mi in r["misses"][:6]:
                print(f"            MISS  {mi['tag']!r:28} trigger={_f(mi['trigger'], '{:.4f}')}")
    print("\n=== all speakers (each enrolled speaker vs its positives + ALL negatives)")
    print(f"  {'rule':8} {'TP':>4} {'FN':>4} {'FP':>4} {'rep':>4} {'recall (95% CI)':>20} "
          f"{'prec':>6} {'FA/h':>6} {'FA/h<=':>7} {'neg min':>8} {'lat med/max ms':>15}")
    for name, t in rep["total"].items():
        lo, hi = t["recall_ci95"]
        print(f"  {name:8} {t['tp']:4} {t['fn']:4} {t['fp']:4} {t['repeats']:4} "
              f"{_f(t['recall'], '{:.2f}'):>6} ({lo:.2f}-{hi:.2f})   "
              f"{_f(t['precision'], '{:.2f}'):>6} {_f(t['fa_per_hour'], '{:.1f}'):>6} "
              f"{_f(t['fa_per_hour_upper95'], '{:.1f}'):>7} {t['negative_minutes']:8.1f} "
              f"{_f(t['latency_ms_median'], '{:.0f}'):>7}/{_f(t['latency_ms_max'], '{:.0f}')}"
              f"   separation {_f(t['separation'], '{:.2f}')}")
    c = rep["cpu"]
    print(f"\nCPU ({c['audio_minutes']:.1f} min audio): features {c['features_ms_per_chunk']:.2f} "
          f"ms/chunk, matching {c['match_ms_per_gated_chunk']:.2f} ms per gated chunk"
          + (f" (legacy {c['legacy_match_ms_per_gated_chunk']:.2f})"
             if c["legacy_match_ms_per_gated_chunk"] else "")
          + f", gate open {100 * c['gate_open_fraction']:.0f}% -> "
          f"~{c['cpu_percent_one_core']:.1f}% of one core"
          + (f" (legacy ~{c['legacy_cpu_percent_one_core']:.1f}%)"
             if "legacy_cpu_percent_one_core" in c else ""))
    print("FA/h<= is the 95% upper bound; with few negative minutes it stays large even at 0 FP.")


# -------------------------------------------------------------------- live / scan ----
class OnlineDetector:
    """Streaming version of the offline pipeline (same Matcher, Rule and run logic)."""

    def __init__(self, templates: list[np.ndarray], threshold: float, rule: Rule):
        self.matcher, self.threshold, self.rule = Matcher(templates), threshold, rule
        self.keep = max(self.matcher.keep, max(round(len(t) * LEGACY_WARP[-1])
                                               for t in templates))
        self.hist: deque = deque(maxlen=self.keep)
        self.rms: deque = deque(maxlen=self.keep)
        self.run = self.cooldown = 0
        self.index = -1
        self.near: deque = deque(maxlen=13)       # recent chunks below threshold (+-0.5 s)

    def step(self, emb: np.ndarray, rms: float) -> dict | None:
        self.index += 1
        self.hist.append(emb)
        self.rms.append(rms)
        if self.cooldown:
            self.cooldown -= 1
            self.run = 0
            return None
        if max(self.rms) < GATE_RMS:
            self.run = 0
            self.near.append(False)
            return None
        h = np.stack(self.hist)
        if self.rule.legacy:
            s, dist, span, best = legacy_score(h, self.matcher.templates), None, None, 0
        else:
            dist, span = self.matcher(h[-self.matcher.keep:])
            order = np.argsort(dist)
            best = int(order[0])
            s = float(dist[order[:self.rule.k]].mean())
            ratio = span[best] / self.matcher.lens[best]
            if self.rule.span and not self.rule.span[0] <= ratio <= self.rule.span[1]:
                s = float("inf")
        below = s <= self.threshold
        self.near.append(below)
        self.run = self.run + 1 if below else 0
        if self.run < self.rule.min_run:
            return None
        self.run, self.cooldown = 0, REFRACTORY
        return {"chunk": self.index, "distance": round(s, 4), "threshold": self.threshold,
                "best_template": best,
                "template_distances": None if dist is None else
                np.round(np.sort(dist), 4).tolist()[:3],
                "run_chunks": self.rule.min_run, "nearby_chunks": int(sum(self.near)),
                "duration_s": None if span is None else round(float(span[best]) * CHUNK_S, 2)}


def _rule_arg(a) -> tuple[Rule, float, list[np.ndarray]]:
    templates, thresholds = load_templates(Path(a.templates))
    rule = RULES[a.rule]
    return rule, a.threshold or thresholds[a.rule], templates


def cmd_listen(a) -> int:
    rule, thr, templates = _rule_arg(a)
    det = OnlineDetector(templates, thr, rule)
    emb = Embedder(threads=a.threads)
    sd = _sd()
    q: queue.Queue = queue.Queue()
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=CHUNK,
                            device=_device(a.device),
                            callback=lambda indata, *_: q.put(indata[:, 0].copy()))
    print(f"listening {a.seconds:.0f} s, rule={rule.name}, threshold={thr:.4f}; Ctrl+C stops."
          " Every candidate is printed; say 'Hey ELARA' and note when you did.")
    saved: list[np.ndarray] = []
    feat_ms, match_ms, cands = [], [], []
    last_loud = None
    wall0, cpu0 = time.monotonic(), time.process_time()
    try:
        with stream:
            while time.monotonic() - wall0 < a.seconds:
                try:
                    chunk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if a.save_wav:
                    saved.append(chunk)
                t0 = time.perf_counter()
                e = emb.push(chunk)
                t1 = time.perf_counter()
                r = ac_rms(chunk)
                if r > GATE_RMS:
                    last_loud = time.monotonic()
                c = det.step(e, r)
                t2 = time.perf_counter()
                feat_ms.append((t1 - t0) * 1000)
                match_ms.append((t2 - t1) * 1000)
                if c is None:
                    continue
                now = time.monotonic() - wall0
                repeat = bool(cands) and now - cands[-1]["t_s"] < REPEAT_S
                c.update(t_s=round(now, 2), repeat=repeat,
                         latency_ms=round((time.monotonic() - last_loud) * 1000)
                         if last_loud else None,
                         compute_ms=round((t2 - t0) * 1000, 1))
                cands.append(c)
                print(f"  {'REPEAT' if repeat else 'WAKE  '} t={c['t_s']:6.2f}s "
                      f"d={c['distance']} thr={thr:.4f} tmpl={c['best_template']} "
                      f"top3={c['template_distances']} run={c['run_chunks']} "
                      f"nearby={c['nearby_chunks']} dur={c['duration_s']}s "
                      f"after-speech={c['latency_ms']}ms compute={c['compute_ms']}ms", flush=True)
    except KeyboardInterrupt:
        pass
    wall, cpu = time.monotonic() - wall0, time.process_time() - cpu0
    print(f"\n{len(feat_ms)} chunks in {wall:.1f} s, candidates: {len(cands)} "
          f"({sum(c['repeat'] for c in cands)} repeats of an earlier event)")
    if feat_ms:
        tot = np.add(feat_ms, match_ms)
        print(f"compute per 80 ms chunk: features median {np.median(feat_ms):.1f} ms, matching "
              f"median {np.median(match_ms):.1f} ms, total p95 {np.percentile(tot, 95):.1f} ms, "
              f"max {tot.max():.1f} ms")
    print(f"process CPU: {100 * cpu / max(wall, 1e-9):.1f}% of one core")
    if a.save_wav and saved:
        write_wav(Path(a.save_wav), np.concatenate(saved))
        print(f"session audio saved to {a.save_wav} (re-analyse with `scan`; delete when done)")
    return 0


def cmd_scan(a) -> int:
    rule, thr, templates = _rule_arg(a)
    emb = Embedder(threads=a.threads)
    matcher = Matcher(templates)
    for f in a.wav:
        e, r, fs = emb.clip(read_wav(Path(f)))
        tr = trace_clip(e, r, matcher, fs, with_legacy=rule.legacy)
        score, best = rule_scores(tr, rule, matcher.lens)
        fires = decide(score, rule, thr)
        print(f"{f}: {len(r) * CHUNK_S:.1f} s, {len(fires)} candidate(s), rule={rule.name}, "
              f"thr={thr:.4f}, lowest trigger score {trigger_score(score, rule):.4f}")
        for fi in fires:
            print("   ", json.dumps(_candidate(score, best, tr, matcher.lens, fi, thr)))
    return 0


# --------------------------------------------------------------------------- main ----
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-models")
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--device")
    c.add_argument("--enroll", type=int, default=5)
    c.add_argument("--positives", type=int, default=20)
    c.add_argument("--hard", type=int, default=2, help="takes per hard-negative phrase")
    c.add_argument("--calib-seconds", type=float, default=30)
    c.add_argument("--calib-hard", type=int, default=1, choices=(0, 1),
                   help="also record each hard negative once for calibration (not for test)")
    c.add_argument("--conversation-seconds", type=float, default=90)
    c.add_argument("--tv-seconds", type=float, default=180)
    c.add_argument("--silence-seconds", type=float, default=60)
    s = sub.add_parser("synth")
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--conversation-seconds", type=float, default=120)
    s.add_argument("--tv-seconds", type=float, default=300)
    s.add_argument("--silence-seconds", type=float, default=60)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--set", required=True)
    ev.add_argument("--save-templates", help="write templates + calibrated thresholds (.npz)")
    ev.add_argument("--no-legacy", action="store_true", help="skip the slow original scorer")
    for sp in (ev, ls := sub.add_parser("listen"), sc := sub.add_parser("scan")):
        sp.add_argument("--threads", type=int, default=1, help="onnxruntime threads (default 1)")
    for sp in (ls, sc):
        sp.add_argument("--templates", default=str(DEFAULT_TEMPLATES))
        sp.add_argument("--rule", choices=list(RULES), default="robust")
        sp.add_argument("--threshold", type=float, help="override the calibrated threshold")
    ev.add_argument("--json", action="store_true")
    ls.add_argument("--seconds", type=float, default=60.0)
    ls.add_argument("--device")
    ls.add_argument("--save-wav", help="keep the session audio for offline `scan` (opt-in)")
    sc.add_argument("wav", nargs="+")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "fetch-models":
        fetch_models()
        return 0
    return {"collect": cmd_collect, "synth": cmd_synth, "evaluate": cmd_evaluate,
            "listen": cmd_listen, "scan": cmd_scan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
