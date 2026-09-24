"""Tests for the wake-word spike's matching logic (no models, no audio device, no network)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import wakeword_bench as ww  # noqa: E402


def unit(rng, n):
    x = rng.normal(size=(n, 96))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_dtw_is_zero_for_identical_and_tolerates_time_stretch():
    rng = np.random.default_rng(0)
    t = unit(rng, 10)
    assert abs(ww.dtw(t, t)) < 1e-12
    assert ww.dtw(np.repeat(t, 2, axis=0), t) < 0.05 < ww.dtw(unit(rng, 10), t)


def test_template_is_cut_around_the_loud_region():
    emb = unit(np.random.default_rng(1), 30)
    rms = np.full(30, 20.0)
    rms[10:20] = 3000.0
    t = ww.phrase_template(emb, rms)
    assert np.array_equal(t, emb[10:23])  # loud chunks 10..19 plus 3 after
    assert ww.phrase_template(emb, np.full(30, 20.0)) is None  # silence: no template


def test_detector_fires_once_then_respects_refractory_and_gate():
    rng = np.random.default_rng(2)
    t = unit(rng, 8)
    det = ww.Detector([t], threshold=0.05)
    noise = unit(rng, 20)
    stream = list(noise) + list(t) + list(t)  # phrase twice back-to-back
    hits = [det.step(e, 3000.0) for e in stream]
    assert sum(h is not None for h in hits) == 1   # second copy is inside the cooldown
    quiet = ww.Detector([t], threshold=0.05)
    assert all(quiet.step(e, 50.0) is None for e in list(t) * 2)  # below the energy gate


def test_calibration_threshold_uses_background_when_separable():
    rng = np.random.default_rng(3)
    base = unit(rng, 8)
    clips = []
    for _ in range(3):
        e = np.concatenate([unit(rng, 5), base + rng.normal(0, 0.05, base.shape), unit(rng, 5)])
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        rms = np.r_[np.full(5, 20.0), np.full(8, 3000.0), np.full(5, 20.0)]
        clips.append((e, rms))
    cal = ww.calibrate(clips, [unit(rng, 40)], margin=1.15)
    assert max(cal["positives"]) < cal["threshold"] < min(cal["negatives"])


def test_chunks_pad_with_edge_value_not_zero():
    pcm = np.full(ww.CHUNK + 10, 7800, np.int16)
    c = ww._chunks(pcm)
    assert c.shape == (2, ww.CHUNK) and ww.ac_rms(c[-1]) == 0.0
