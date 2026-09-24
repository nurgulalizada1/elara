import math
import shutil
from array import array

import pytest

from elara.voice import (
    EnergyVAD,
    Transcript,
    TranscriptQualityGate,
    UtteranceSegmenter,
    VoicePipeline,
)
from elara.voice.interfaces import FRAME_BYTES
from elara.voice.tts import EspeakTTS, TTSUnavailable, build_tts


def tone(amplitude: int) -> bytes:
    n = FRAME_BYTES // 2
    return array("h", [int(amplitude * math.sin(2 * math.pi * 440 * i / 16000))
                       for i in range(n)]).tobytes()


SILENCE, SPEECH = tone(20), tone(8000)


def test_vad_distinguishes_speech_from_silence():
    vad = EnergyVAD()
    assert not any(vad.is_speech(SILENCE) for _ in range(10))
    assert vad.is_speech(SPEECH)


def test_segmenter_emits_one_utterance():
    seg = UtteranceSegmenter(EnergyVAD(), silence_ms=150, min_speech_ms=90)
    frames = [SILENCE] * 5 + [SPEECH] * 10 + [SILENCE] * 10
    out = [pcm for f in frames if (pcm := seg.push(f))]
    assert len(out) == 1 and len(out[0]) >= 10 * FRAME_BYTES


def test_segmenter_drops_clicks():
    seg = UtteranceSegmenter(EnergyVAD(), silence_ms=150, min_speech_ms=300)
    assert not [p for f in [SPEECH] + [SILENCE] * 10 if (p := seg.push(f))]


@pytest.mark.parametrize("tr,ok", [
    (Transcript("Salam ELARA, bu gün hava necədir?", confidence=0.9, no_speech_prob=0.1), True),
    (Transcript("", confidence=0.9), False),
    (Transcript("Thank you.", confidence=0.9), False),
    (Transcript("hello", confidence=0.2), False),
    (Transcript("hello there", no_speech_prob=0.9), False),
    (Transcript("the the the the the the the the", confidence=0.9), False),
])
def test_quality_gate(tr, ok):
    assert TranscriptQualityGate().check(tr).accepted is ok


class FakeSTT:
    name = "fake"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, pcm, language=None):
        self.calls += 1
        return Transcript(self.text, "az", 0.9, 0.05, len(pcm) / 32000)


class FakeWake:
    phrase = "hey elara"

    def __init__(self):
        self.armed = False

    def detect(self, frame):
        return frame == b"WAKE" + b"\0" * (FRAME_BYTES - 4)


async def test_pipeline_wake_then_utterance():
    wake_frame = b"WAKE" + b"\0" * (FRAME_BYTES - 4)
    replies = []

    async def respond(text):
        replies.append(text)
        return "Salam!", "az"

    stt = FakeSTT("Salam ELARA")
    p = VoicePipeline(segmenter=UtteranceSegmenter(EnergyVAD(), silence_ms=90, min_speech_ms=60),
                      stt=stt, respond=respond, wake=FakeWake())
    # speech before the wake word is ignored entirely (never transcribed)
    for f in [SPEECH] * 5 + [SILENCE] * 5:
        assert await p.feed(f) is None
    assert stt.calls == 0
    await p.feed(wake_frame)
    turns = [t for f in [SPEECH] * 15 + [SILENCE] * 5 if (t := await p.feed(f))]
    assert len(turns) == 1, turns
    assert turns[0].accepted, turns[0].reason
    assert turns[0].reply_text == "Salam!" and replies == ["Salam ELARA"]
    assert p.state == "waiting_for_wake"


async def test_pipeline_rejects_bad_transcript():
    async def respond(text):
        raise AssertionError("must not be called")

    p = VoicePipeline(segmenter=UtteranceSegmenter(EnergyVAD(), silence_ms=90, min_speech_ms=60),
                      stt=FakeSTT("Thank you."), respond=respond)
    turns = [t for f in [SPEECH] * 5 + [SILENCE] * 5 if (t := await p.feed(f))]
    assert turns and not turns[0].accepted


def test_tts_registry_and_missing_binary(monkeypatch):
    with pytest.raises(TTSUnavailable):
        build_tts("nope", {})
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(TTSUnavailable, match="not installed"):
        EspeakTTS().synthesize("x", "az")


@pytest.mark.skipif(not (shutil.which("espeak-ng") or shutil.which("espeak")),
                    reason="espeak-ng not installed")
def test_espeak_produces_wav():
    assert EspeakTTS().synthesize("Salam", "az").startswith(b"RIFF")
