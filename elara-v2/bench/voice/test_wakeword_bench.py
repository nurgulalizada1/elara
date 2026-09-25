"""Tests for the wake-word spike's matching/decision logic (no audio device, no network).

Tests that need the two ONNX feature models skip when `fetch-models` has not been run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import wakeword_bench as ww  # noqa: E402


def unit(rng, n):
    x = rng.normal(size=(n, 96))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_subsequence_dtw_finds_the_phrase_ending_now_and_its_start():
    rng = np.random.default_rng(0)
    t = unit(rng, 8)
    hist = np.concatenate([unit(rng, 6), t])
    d, start = ww.subseq_dtw((1.0 - t @ hist.T).tolist())
    assert abs(d) < 1e-9 and start == 6
    slow = np.concatenate([unit(rng, 4), np.repeat(t, 2, axis=0)])  # spoken twice as slowly
    d_slow, start_slow = ww.subseq_dtw((1.0 - t @ slow.T).tolist())
    assert d_slow < 0.05 and start_slow in (4, 5)  # either copy of the first frame
    d_other, _ = ww.subseq_dtw((1.0 - t @ unit(rng, 14).T).tolist())
    assert d_other > 0.5


def test_matcher_reports_distance_and_span_per_template():
    rng = np.random.default_rng(1)
    t1, t2 = unit(rng, 8), unit(rng, 10)
    m = ww.Matcher([t1, t2])
    dist, span = m(np.concatenate([unit(rng, 10), t1]))
    assert dist[0] < 1e-9 < dist[1] and span[0] == 8
    assert abs(ww.dtw(t1, t1)) < 1e-12  # legacy scorer kept for comparison


def test_template_is_cut_around_the_loud_region():
    emb = unit(np.random.default_rng(2), 30)
    rms = np.full(30, 20.0)
    rms[10:20] = 3000.0
    assert np.array_equal(ww.phrase_template(emb, rms), emb[10:23])  # + 3 chunks after
    assert ww.phrase_template(emb, np.full(30, 20.0)) is None
    assert ww.truth_window(rms) == (10, 19, 19 + ww.TRUTH_TAIL)


def test_decide_requires_a_run_and_respects_the_refractory_period():
    score = np.array([0.9, 0.01, 0.9, 0.01, 0.01, 0.01, 0.01] + [0.01] * 30)
    single = ww.decide(score, ww.Rule("x"), 0.05)
    assert single[0] == 1 and all(b - a > ww.REFRACTORY for a, b in zip(single, single[1:], strict=False))
    sustained = ww.decide(score, ww.Rule("y", min_run=2), 0.05)
    assert sustained[0] == 4                      # the isolated dip at chunk 1 is ignored
    assert ww.trigger_score(score, ww.Rule("y", min_run=2)) == pytest.approx(0.01)
    assert ww.trigger_score(np.array([0.9, 0.01, 0.9]), ww.Rule("y", min_run=2)) == 0.9


def test_rule_scores_aggregate_templates_and_check_duration():
    tr = ww.Trace(rms=np.ones(2), gate=np.ones(2, bool), legacy=np.array([0.3, 0.4]),
                  dist=np.array([[0.1, 0.3, 0.5], [0.2, 0.2, 0.2]]),
                  span=np.array([[10, 10, 10], [30, 10, 10]]), feature_s=0, match_s=0,
                  legacy_s=0)
    lens = np.array([10, 10, 10])
    s, best = ww.rule_scores(tr, ww.Rule("k2", k=2), lens)
    assert s == pytest.approx([0.2, 0.2]) and best[0] == 0
    s, _ = ww.rule_scores(tr, ww.Rule("span", span=(0.6, 1.6)), lens)
    assert s[0] == pytest.approx(0.1) and np.isinf(s[1])   # 30 chunks for a 10-chunk template
    assert list(ww.rule_scores(tr, ww.RULES["legacy"], lens)[0]) == [0.3, 0.4]


def test_classification_counts_repeats_and_false_activations():
    window = (10, 20, 32)
    c = ww._classify([21, 30, 60], positive=True, window=window)
    assert (c["tp"], c["repeat"], c["fp"]) == (1, 1, 1)
    assert c["latency_ms"] == pytest.approx(80.0)
    assert ww._classify([5], positive=False, window=(0, 0, 0))["fp"] == 1
    s = ww.summarize([{**c, "positive": True, "seconds": 3.0, "trigger": 0.02},
                      {"tp": 0, "fp": 0, "repeat": 0, "latency_ms": None, "positive": False,
                       "seconds": 3600.0, "trigger": 0.2}])
    assert (s["tp"], s["fp"], s["fn"], s["repeats"]) == (1, 1, 0, 1)
    assert s["fa_per_hour"] == pytest.approx(1.0) and s["separation"] == pytest.approx(10.0)


def test_chunks_pad_with_edge_value_not_zero():
    c = ww._chunks(np.full(ww.CHUNK + 10, 7800, np.int16))
    assert c.shape == (2, ww.CHUNK) and ww.ac_rms(c[-1]) == 0.0


@pytest.mark.skipif(not all((ww.MODEL_DIR / n).exists() for n in ww.FEATURE_MODELS),
                    reason="feature models not fetched")
def test_incremental_mel_matches_full_recompute():
    e = ww.Embedder()
    x = np.random.default_rng(3).normal(0, 2000, 12_960).astype(np.float32)
    full = e._mel(x)
    assert np.allclose(full[-8:], e._mel(x[-ww.MEL_NEW:]), atol=1e-5)
