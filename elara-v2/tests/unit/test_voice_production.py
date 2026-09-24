"""Production English voice channel: config, microphone, STT, gate, TTS, voice -> ElaraCore.

Hardware-free: the microphone stream, faster-whisper model and espeak binary are faked.
"""

import math
import threading
import time
import types
from array import array

import pytest

from elara.cli import main as cli
from elara.core.service import ElaraCore
from elara.voice.config import FRAME_MS, SAMPLE_RATE, VoiceConfig
from elara.voice.interfaces import Transcript
from elara.voice.microphone import Microphone, MicrophoneError
from elara.voice.quality import TranscriptQualityGate
from elara.voice.session import VoiceSession
from elara.voice.stt import FasterWhisperSTT, STTUnavailable
from elara.voice.tts import EspeakSpeaker, TTSUnavailable
from elara.voice.vad import SpeechEndpointer
from tests.fakes import ScriptedProvider, text

np = pytest.importorskip("numpy")
N = SAMPLE_RATE * FRAME_MS // 1000  # samples per frame


def frame(amp: int) -> bytes:
    return array("h", [int(amp * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
                       for i in range(N)]).tobytes()


SIL, SPEECH = frame(30), frame(8000)


class FakeStream:
    """Pushes frames to the callback from a thread, like PortAudio does."""

    def __init__(self, on_frame, frames):
        self.on_frame, self.frames = on_frame, frames
        self.started = self.stopped = self.closed = False
        self.overflow_counter = {"n": 0}

    def start(self):
        self.started = True
        threading.Thread(target=lambda: [self.on_frame(f) for f in self.frames],
                         daemon=True).start()

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def mic(frames, **cfg):
    streams = []

    def factory(on_frame):
        streams.append(FakeStream(on_frame, frames))
        return streams[-1]

    config = VoiceConfig(**{"start_timeout_s": 1.0, "end_silence_ms": 300, **cfg})
    return Microphone(config, stream_factory=factory), streams


# ------------------------------------------------------------------ configuration ----
def test_config_defaults_and_english_only(monkeypatch):
    c = VoiceConfig()
    assert (c.language, c.model, c.device, c.compute_type, c.cpu_threads, c.beam_size) == \
        ("en", "small", "cpu", "int8", 8, 5)
    monkeypatch.setenv("ELARA_VOICE_CPU_THREADS", "6")
    assert VoiceConfig().cpu_threads == 6
    for bad in ({"language": "az"}, {"language": "tr"}, {"device": "cuda"}):
        with pytest.raises(Exception):
            VoiceConfig(**bad)
    monkeypatch.setenv("ELARA_VOICE_LANGUAGE", "az")
    with pytest.raises(Exception):
        VoiceConfig()


# --------------------------------------------------------------------- microphone ----
_rng = __import__("random").Random(1234)


def noise(rms: float) -> bytes:
    """Gaussian background noise at a given int16 RMS (deterministic seed)."""
    return array("h", [max(-32768, min(32767, int(_rng.gauss(0, rms)))) for _ in range(N)]
                 ).tobytes()


def voiced(level: float, bg: float = 0.0) -> bytes:
    """Speech-like frame (tone at RMS `level`) mixed with background noise `bg`."""
    return array("h", [max(-32768, min(32767, int(level * 1.414 * math.sin(
        2 * math.pi * 220 * i / SAMPLE_RATE) + (_rng.gauss(0, bg) if bg else 0))))
        for i in range(N)]).tobytes()


def room(bg: float, seconds: float) -> list[bytes]:
    return [noise(bg) for _ in range(int(seconds * 1000 / FRAME_MS))]


# The tail of every stream is 12 s of room noise: like a real mic it never "runs out",
# so an early end can only come from end-of-speech detection.
def test_speech_then_silence_ends_well_before_max_duration():
    m, streams = mic(room(60, 0.5) + [voiced(6000, 60)] * 50 + room(60, 12))
    u = m.listen()
    assert u.reason == "end_of_speech" and streams[0].stopped and streams[0].closed
    assert 1.5 <= u.duration_s <= 3.0  # ~1.5 s speech + pre-roll + end silence
    assert u.trailing_silence_ms >= 300 and u.speech_ms >= 1400


def test_laptop_mic_background_above_absolute_minimum_still_ends():
    """Regression: background at ~-37 dBFS (RMS 450 > vad_min_rms 300) used to be classified
    as speech forever, so capture always ran to max_duration."""
    m, _ = mic(room(450, 0.5) + [voiced(6000, 450)] * 50 + room(450, 12),
               start_timeout_s=8.0, max_utterance_s=12.0)
    u = m.listen()
    assert u.reason == "end_of_speech" and u.duration_s < 4
    assert -40 < u.noise_floor_dbfs < -34 and u.start_threshold_dbfs > u.stop_threshold_dbfs


def test_continuous_speech_with_word_gaps_is_not_cut():
    words = [voiced(6000, 80)] * 12
    gap = room(80, 0.3)  # 300 ms pauses between phrases, shorter than end_silence (800 ms)
    m, _ = mic(room(80, 0.5) + words + gap + words + gap + words + room(80, 12))
    u = m.listen()
    assert u.reason == "end_of_speech"
    assert u.duration_s >= 36 * FRAME_MS / 1000 + 0.6  # all three phrases + both gaps kept


def test_noisy_but_spoken_input_does_not_end_immediately():
    # loud fan (~-31 dBFS) with speech ~16 dB above it; speech level also varies
    speech = [voiced(lvl, 900) for lvl in [5000, 6500, 3500, 7000] * 15]
    m, _ = mic(room(900, 0.5) + speech + room(900, 12))
    u = m.listen()
    assert u.reason == "end_of_speech"
    assert u.speech_ms >= 1500 and u.duration_s >= 1.8  # not truncated after a few frames


def test_short_click_before_speech_is_ignored():
    m, _ = mic(room(60, 0.5) + [voiced(20000)] * 2 + room(60, 0.5)
               + [voiced(6000, 60)] * 30 + room(60, 12))
    u = m.listen()
    assert u.reason == "end_of_speech" and u.speech_ms >= 30 * FRAME_MS - 100
    m, _ = mic(room(60, 0.5) + [voiced(20000)] * 2 + room(60, 12), start_timeout_s=1.5)
    assert m.listen().reason == "no_speech"  # a click alone never becomes an utterance


def test_initial_silence_still_times_out():
    m, streams = mic(room(60, 12), start_timeout_s=1.0)
    t0 = time.monotonic()
    u = m.listen()
    assert u.pcm == b"" and u.reason == "no_speech" and streams[0].closed
    assert time.monotonic() - t0 < 3


def test_max_duration_remains_the_safety_cap():
    m, _ = mic(room(60, 0.5) + [voiced(6000, 60)] * 400, max_utterance_s=2.0)
    u = m.listen()
    assert u.reason == "max_duration" and 1.9 <= u.duration_s <= 2.4


def test_endpointer_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("ELARA_VOICE_END_SILENCE_MS", "400")
    monkeypatch.setenv("ELARA_VOICE_VAD_STOP_RATIO", "1.5")
    monkeypatch.setenv("ELARA_VOICE_CALIBRATION_MS", "150")
    c = VoiceConfig()
    assert (c.end_silence_ms, c.vad_stop_ratio, c.calibration_ms) == (400, 1.5, 150)
    with pytest.raises(ValueError):
        SpeechEndpointer(start_ratio=2.0, stop_ratio=3.0)


def test_microphone_errors_are_reported(monkeypatch):
    def broken(on_frame):
        s = FakeStream(on_frame, [])
        s.start = lambda: (_ for _ in ()).throw(OSError("device busy"))
        return s

    with pytest.raises(MicrophoneError, match="cannot start microphone"):
        Microphone(VoiceConfig(), stream_factory=broken).listen()
    import elara.voice.microphone as micmod

    def no_portaudio():
        raise MicrophoneError("PortAudio is not available (sudo apt install libportaudio2)")

    monkeypatch.setattr(micmod, "_sounddevice", no_portaudio)
    with pytest.raises(MicrophoneError, match="libportaudio2"):
        Microphone(VoiceConfig()).listen()


# ---------------------------------------------------------------------------- STT ----
class FakeWhisperModel:
    loads = 0

    def __init__(self, size, **kw):
        FakeWhisperModel.loads += 1
        self.size, self.kw, self.calls = size, kw, []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        seg = types.SimpleNamespace(text=" What time is it? ", avg_logprob=-0.25,
                                    no_speech_prob=0.02, compression_ratio=0.9)
        return iter([seg]), types.SimpleNamespace(language=kw["language"],
                                                  language_probability=1.0)


def test_stt_loads_once_forces_english_and_reports_metrics():
    FakeWhisperModel.loads = 0
    stt = FasterWhisperSTT(VoiceConfig(), model_factory=FakeWhisperModel)
    pcm = SPEECH * 50  # 1.5 s
    t1, t2 = stt.transcribe(pcm), stt.transcribe(pcm)
    assert FakeWhisperModel.loads == 1
    m = stt._model
    assert m.size == "small" and m.kw["device"] == "cpu" and m.kw["compute_type"] == "int8"
    assert m.kw["cpu_threads"] == 8 and m.kw["num_workers"] == 1
    assert all(c["language"] == "en" and c["beam_size"] == 5 for c in m.calls)
    assert t1.text == "What time is it?" and t1.language == "en"
    assert (t1.language_probability, t1.avg_logprob, t1.no_speech_prob,
            t1.compression_ratio) == (1.0, -0.25, 0.02, 0.9)
    assert t1.duration_s == 1.5 and t1.real_time_factor == round(
        t1.transcription_time_s / 1.5, 3)
    assert t1.confidence is None  # not invented: faster-whisper has no confidence field
    assert t2.text == t1.text


def test_stt_refuses_non_english_production_language():
    stt = FasterWhisperSTT(VoiceConfig(), model_factory=FakeWhisperModel)
    for lang in ("az", "tr"):
        with pytest.raises(ValueError, match="not a production voice language"):
            stt.transcribe(SPEECH, language=lang)
    assert stt.transcribe(SPEECH, language="en").language == "en"


def test_stt_failures():
    def boom(*a, **k):
        raise RuntimeError("model download blocked")

    with pytest.raises(STTUnavailable, match="cannot load Whisper model"):
        FasterWhisperSTT(VoiceConfig(), model_factory=boom).load()

    class Crashing(FakeWhisperModel):
        def transcribe(self, audio, **kw):
            raise RuntimeError("decoder crashed")

    tr = FasterWhisperSTT(VoiceConfig(), model_factory=Crashing).transcribe(SPEECH * 10)
    assert tr.text == "" and "decoder crashed" in tr.error


# --------------------------------------------------------------------- quality gate --
GATE = TranscriptQualityGate(expected_language="en")


def good(text, **kw):
    return Transcript(**{"text": text, "language": "en", "duration_s": 1.2,
                         "avg_logprob": -0.3, "no_speech_prob": 0.03,
                         "compression_ratio": 1.0, **kw})


@pytest.mark.parametrize("utterance", ["stop", "cancel", "open calendar", "what time is it",
                                       "Stop.", "thank you", "What's 17 times 42?"])
def test_gate_accepts_normal_short_commands(utterance):
    r = GATE.check(good(utterance))
    assert r.accepted and r.code == "ok", r


@pytest.mark.parametrize("tr,code", [
    (good("", error="RuntimeError: boom"), "stt_failed"),
    (good(""), "empty"),
    (good("... ♪"), "empty"),
    (good("stop", duration_s=0.1), "too_short_audio"),
    (good("hello", no_speech_prob=0.9, avg_logprob=-1.4), "no_speech"),
    (good("hello there", avg_logprob=-1.6), "low_logprob"),
    (good("the the the the the the the the"), "repetitive"),
    (good("open calendar", compression_ratio=3.1), "repetitive"),
    (good("um"), "filler"),
    (good("Uh, hmm."), "filler"),
    (good("Thank you.", no_speech_prob=0.5), "hallucination"),
    (good("Thanks for watching!", avg_logprob=-0.8), "hallucination"),
    (good("salam", language="az"), "wrong_language"),
])
def test_gate_rejects_unusable_output(tr, code):
    r = GATE.check(tr)
    assert not r.accepted and r.code == code and r.reason


def test_gate_is_deterministic():
    tr = good("open calendar")
    assert len({(GATE.check(tr).accepted, GATE.check(tr).code) for _ in range(10)}) == 1


# ----------------------------------------------------------------------------- TTS ---
def test_espeak_speaker_is_english_and_replaceable(tmp_path):
    script = tmp_path / "espeak-ng"
    script.write_text("#!/bin/sh\necho \"$@\" > \"$(dirname \"$0\")/args\"\n"
                      "printf 'RIFF....WAVEfmt '\n")
    script.chmod(0o755)
    played = []
    sp = EspeakSpeaker(voice="en-us", binary=str(script), player=played.append, max_chars=20)
    assert sp.available()
    sp.speak("The time is 10:42 and this sentence is long")
    args = (tmp_path / "args").read_text()
    assert "-v en-us" in args and "…" in args and played and played[0].startswith(b"RIFF")


def test_speaker_unavailable_raises():
    with pytest.raises(TTSUnavailable):
        EspeakSpeaker(binary="/nonexistent/espeak-ng", player=lambda b: None).speak("hi")


# ----------------------------------------------------- voice -> ElaraCore round trip --
class FakeSTT:
    language = "en"

    def __init__(self, transcript):
        self.transcript, self.seen = transcript, []

    def transcribe(self, pcm):
        self.seen.append(len(pcm))
        return self.transcript


class FakeMic:
    def __init__(self, pcm=SPEECH * 40):
        self.pcm = pcm

    def listen(self):
        from elara.voice.microphone import Utterance
        return Utterance(pcm=self.pcm, duration_s=len(self.pcm) / 32000, capture_s=1.5,
                         reason="end_of_speech" if self.pcm else "no_speech")


class RecordingSpeaker:
    name = "fake"

    def __init__(self, fail=False):
        self.said, self.fail = [], fail

    def available(self):
        return True

    def speak(self, text):
        if self.fail:
            raise TTSUnavailable("no audio output")
        self.said.append(text)
        return 0.1


@pytest.fixture
def core(settings):
    c = ElaraCore.open(settings.model_copy(update={"model_fast": "f", "model_strong": "s"}),
                       provider=ScriptedProvider())
    yield c


async def test_voice_turn_reaches_core_as_voice_channel_without_llm(core):
    speaker = RecordingSpeaker()
    session = VoiceSession(core, FakeMic(), FakeSTT(good("What time is it?")),
                           speaker=speaker)
    seen = []
    real = core.process

    async def spy(req):
        seen.append(req)
        return await real(req)

    core.process = spy
    turn = await session.listen_once()
    assert turn.status == "answered" and seen[0].channel == "voice"
    assert seen[0].text == "What time is it?"  # exactly the transcript, nothing added
    r = turn.result
    assert r.channel == "voice" and r.intent == "time" and r.used_llm is False
    assert r.resolver == "deterministic" and core.container.llm.provider.requests == []
    assert speaker.said == [r.text] and turn.spoken
    assert turn.utterance.pcm == b""  # raw audio dropped after transcription
    assert {"capture_s", "audio_s", "stt_s", "rtf", "core_s", "speak_s", "total_s"} <= \
        turn.timings.keys()


async def test_rejected_transcript_never_reaches_core(core):
    session = VoiceSession(core, FakeMic(), FakeSTT(good("um")))
    turn = await session.listen_once()
    assert turn.status == "rejected" and turn.gate.code == "filler" and turn.result is None
    with core.container.db.connect() as c:
        assert c.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


async def test_no_speech_skips_stt(core):
    stt = FakeSTT(good("hello"))
    turn = await VoiceSession(core, FakeMic(pcm=b""), stt).listen_once()
    assert turn.status == "no_speech" and stt.seen == [] and turn.transcript is None


async def test_llm_request_and_conversation_context(core):
    core.container.llm.provider.push(text("Paris."), text("About two million."))
    session = VoiceSession(core, FakeMic(), FakeSTT(good("What is the capital of France?")))
    t1 = await session.listen_once()
    session.stt = FakeSTT(good("And its population?"))
    t2 = await session.listen_once()
    assert t1.result.used_llm and t2.result.conversation_id == t1.result.conversation_id
    assert [m.text for m in core.container.llm.provider.requests[1].messages][:2] == \
        ["What is the capital of France?", "Paris."]


async def test_voice_transcript_gets_same_security_as_text(core, settings):
    (settings.write_dirs[0] / "keep.txt").write_text("original")
    session = VoiceSession(core, FakeMic(), FakeSTT(good("Read /etc/hostname")))
    t = await session.listen_once()
    assert t.result.tool_calls[0].status == "denied" and not t.result.used_llm
    # destructive actions still need explicit confirmation
    core.container.llm.provider.push(
        __import__("tests.fakes", fromlist=["tool"]).tool("delete_file", {"path": "keep.txt"}))
    session.stt = FakeSTT(good("Delete keep.txt from my workspace"))
    t = await session.listen_once()
    assert t.result.pending_action is not None
    assert (settings.write_dirs[0] / "keep.txt").read_text() == "original"


async def test_speaker_failure_keeps_text_answer(core):
    session = VoiceSession(core, FakeMic(), FakeSTT(good("what time is it")),
                           speaker=RecordingSpeaker(fail=True))
    t = await session.listen_once()
    assert t.result is not None and not t.spoken and "no audio output" in t.speak_error


def test_no_azerbaijani_production_voice_path():
    import inspect

    import elara.voice.session as session_mod
    import elara.voice.stt as stt_mod
    assert VoiceConfig.model_fields["language"].annotation.__args__ == ("en",)
    assert '"az"' not in inspect.getsource(session_mod)
    assert "language=self.language" in inspect.getsource(stt_mod)


# ------------------------------------------------------------------ CLI (dev mode) ---
def test_cli_voice_reports_missing_microphone(monkeypatch, settings, capsys):
    import elara.voice.microphone as micmod

    def no_portaudio():
        raise MicrophoneError("PortAudio is not available (sudo apt install libportaudio2)")

    monkeypatch.setattr(micmod, "_sounddevice", no_portaudio)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    assert cli.main(["voice", "--list-devices"]) == 2
    assert "libportaudio2" in capsys.readouterr().err


def test_cli_voice_one_bounded_turn(monkeypatch, settings, capsys):
    import elara.cli.voice_cmd as vc
    import elara.voice.microphone as micmod
    import elara.voice.stt as sttmod

    class OneShotMic(FakeMic):
        def __init__(self, config, stream_factory=None):
            super().__init__()

    monkeypatch.setattr(micmod, "Microphone", OneShotMic)
    monkeypatch.setattr(sttmod, "FasterWhisperSTT",
                        lambda cfg: FasterWhisperSTT(cfg, model_factory=FakeWhisperModel))
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    assert vc  # imported for monkeypatch side effects
    assert cli.main(["voice", "--no-speak"]) == 0
    out = capsys.readouterr().out
    assert "heard:   'What time is it?'" in out and "gate:    accepted" in out
    assert "resolver=deterministic" in out and "used_llm=False" in out
    assert "stt_s=" in out and "core_s=" in out and "total_s=" in out
