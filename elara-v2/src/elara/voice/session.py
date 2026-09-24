"""VoiceSession: the voice channel on top of ElaraCore.

    microphone -> STT (local) -> quality gate -> CoreRequest(channel="voice") -> CoreResult
                                                                              -> speaker

This module adds no routing, tools, memory, confirmations or LLM logic: an accepted
transcript enters ElaraCore exactly like typed text (same injection defences, permission
checks and confirmation flow). Raw audio stays in memory for one utterance and is dropped
after transcription; it is never logged or persisted.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from elara.core.logging import get_logger
from elara.core.service import CoreRequest, CoreResult, ElaraCore
from elara.voice.interfaces import Transcript
from elara.voice.microphone import Microphone, Utterance
from elara.voice.quality import GateResult, TranscriptQualityGate
from elara.voice.stt import FasterWhisperSTT
from elara.voice.tts import Speaker

log = get_logger(__name__)


@dataclass
class VoiceTurn:
    utterance: Utterance | None = None
    transcript: Transcript | None = None
    gate: GateResult | None = None
    result: CoreResult | None = None
    spoken: bool = False
    speak_error: str | None = None
    timings: dict[str, float | None] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.result is not None:
            return "answered"
        if self.gate is not None and not self.gate.accepted:
            return "rejected"
        return "no_speech"


class VoiceSession:
    def __init__(self, core: ElaraCore, microphone: Microphone, stt: FasterWhisperSTT,
                 gate: TranscriptQualityGate | None = None, speaker: Speaker | None = None):
        self.core = core
        self.microphone = microphone
        self.stt = stt
        self.gate = gate or TranscriptQualityGate(expected_language=stt.language)
        self.speaker = speaker
        self.conversation_id: str | None = None

    async def listen_once(self) -> VoiceTurn:
        turn = VoiceTurn()
        t0 = time.perf_counter()
        utt = await asyncio.to_thread(self.microphone.listen)
        turn.utterance = utt
        turn.timings["capture_s"] = utt.capture_s
        turn.timings["audio_s"] = utt.duration_s
        if not utt.pcm:
            turn.timings["total_s"] = round(time.perf_counter() - t0, 3)
            return turn
        pcm, utt.pcm = utt.pcm, b""  # the turn keeps metadata only, never the audio
        turn.transcript = await self.process_audio(pcm, turn)
        turn.timings["total_s"] = round(time.perf_counter() - t0, 3)
        self._log(turn)
        return turn

    async def process_audio(self, pcm: bytes, turn: VoiceTurn) -> Transcript:
        """STT -> gate -> core -> speaker for one captured utterance."""
        transcript = await asyncio.to_thread(self.stt.transcribe, pcm)
        turn.timings["stt_s"] = transcript.transcription_time_s
        turn.timings["rtf"] = transcript.real_time_factor
        turn.gate = self.gate.check(transcript)
        if not turn.gate.accepted:
            return transcript
        turn.result = await self.submit(transcript.text)
        turn.timings["core_s"] = round(turn.result.duration_ms / 1000, 3)
        if self.speaker is not None and turn.result.text:
            try:
                turn.timings["speak_s"] = await asyncio.to_thread(self.speaker.speak,
                                                                  turn.result.text)
                turn.spoken = True
            except Exception as e:  # speech output must never lose the text answer
                turn.speak_error = f"{type(e).__name__}: {e}"
        return transcript

    async def submit(self, text: str) -> CoreResult:
        """Send an accepted transcript to ElaraCore as ordinary user text."""
        result = await self.core.process(CoreRequest(text=text, channel="voice",
                                                     conversation_id=self.conversation_id))
        self.conversation_id = result.conversation_id  # keep context across turns
        return result

    @staticmethod
    def _log(turn: VoiceTurn) -> None:
        # Timing and outcome only: no audio and no transcript text in logs.
        log.info("voice.turn", extra={
            "status": turn.status, "gate": turn.gate.code if turn.gate else None,
            "chars": len(turn.transcript.text) if turn.transcript else 0,
            "resolver": turn.result.resolver if turn.result else None,
            "used_llm": turn.result.used_llm if turn.result else None, **turn.timings})
