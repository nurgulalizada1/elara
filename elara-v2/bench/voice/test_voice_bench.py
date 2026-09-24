"""Tests for the isolated benchmark harness (fake audio device and fake model; no network)."""

import math
import sys
import types
from array import array
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import voice_bench as vb  # noqa: E402


def fake_sounddevice(signal: np.ndarray):
    sd = types.ModuleType("sounddevice")
    sd.default = types.SimpleNamespace(device=(1, 3))
    sd.query_hostapis = lambda: [{"name": "ALSA"}]
    sd.query_devices = lambda: [
        {"name": "HDMI out", "max_input_channels": 0, "hostapi": 0, "default_samplerate": 48000.0},
        {"name": "ALC256 Analog", "max_input_channels": 2, "hostapi": 0, "default_samplerate": 48000.0},
        {"name": "Digital Microphone", "max_input_channels": 4, "hostapi": 0, "default_samplerate": 48000.0}]
    sd.rec = lambda frames, **kw: signal[:frames].reshape(-1, 1)
    sd.wait = lambda: None
    return sd


def tone(seconds=5, amp=8000):
    t = np.arange(int(seconds * vb.SAMPLE_RATE))
    return (amp * np.sin(2 * math.pi * 220 * t / vb.SAMPLE_RATE)).astype(np.int16)


def test_devices(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(tone()))
    info = vb.list_devices()
    assert [d["name"] for d in info["inputs"]] == ["ALC256 Analog", "Digital Microphone"]
    assert info["default_input"] == 1 and info["inputs"][0]["is_default"]


def test_mic_check_writes_16k_mono_and_reports_levels(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(tone()))
    r = vb.mic_check(5, None, tmp_path / "m.wav")
    assert r["exists"] and r["format_ok_16k_mono_16bit"] and r["duration_s"] == 5.0
    assert r["sample_rate"] == 16000 and r["channels"] == 1 and r["sample_width_bytes"] == 2
    assert -20 < r["peak_dbfs"] < -10 and r["verdict"].startswith("OK")


@pytest.mark.parametrize("amp,verdict", [(0, "digital silence"), (100, "very quiet"),
                                         (32767, "clipping")])
def test_level_verdicts(amp, verdict):
    assert vb.levels(array("h", [amp, -amp] * 1000))["verdict"].startswith(verdict)


def test_duration_validation(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(tone()))
    with pytest.raises(ValueError):
        vb.record(0.1, tmp_path / "x.wav")


def test_missing_portaudio_gives_actionable_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real(name, *a, **k)

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit, match="libportaudio2"):
        vb._sd()


def test_transcribe_reports_metrics_with_fake_model(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice(tone(3)))
    wav = vb.record(3, tmp_path / "c.wav")
    seen = {}

    class FakeModel:
        def __init__(self, size, **kw):
            seen["init"] = (size, kw)

        def transcribe(self, audio, **kw):
            seen["transcribe"] = kw
            seen["audio"] = audio
            seg = types.SimpleNamespace(start=0.0, end=2.5, text=" Salam dünya ",
                                        avg_logprob=-0.2, no_speech_prob=0.01)
            info = types.SimpleNamespace(language="az", language_probability=1.0)
            return iter([seg]), info

    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    r = vb.transcribe(wav, "small", 8, 5, "az", local_only=True)
    size, kw = seen["init"]
    assert size == "small" and kw["device"] == "cpu" and kw["compute_type"] == "int8"
    assert kw["cpu_threads"] == 8 and kw["local_files_only"] is True
    assert seen["transcribe"]["language"] == "az" and seen["audio"].dtype == np.float32
    assert r["text"] == "Salam dünya" and r["audio_duration_s"] == 3.0
    assert r["real_time_factor"] == round(r["transcription_time_s"] / 3.0, 3)
    assert {"load_time_s", "language", "cpu_threads", "warnings"} <= r.keys()


def test_cli_rejects_bad_arguments(tmp_path):
    with pytest.raises(SystemExit):
        vb.main(["transcribe", str(tmp_path / "missing.wav"), "--model", "small"])
    with pytest.raises(SystemExit):
        vb.main(["transcribe", "x.wav", "--model", "gigantic"])
