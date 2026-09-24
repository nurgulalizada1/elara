"""Voice pipeline: frames -> [wake word] -> VAD segmentation -> STT -> quality gate ->
Assistant -> TTS. Components are injected, so each can be swapped or tested alone.
Microphone capture/playback are not implemented yet; feed frames from any source.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from elara.core.logging import get_logger
from elara.voice.interfaces import SpeechToText, TextToSpeech, WakeWordDetector
from elara.voice.quality import TranscriptQualityGate
from elara.voice.vad import UtteranceSegmenter

log = get_logger(__name__)


class State(StrEnum):
    WAITING_FOR_WAKE = "waiting_for_wake"
    LISTENING = "listening"


@dataclass
class VoiceTurn:
    transcript: str
    accepted: bool
    reason: str
    reply_text: str | None = None
    reply_audio: bytes | None = None
    language: str | None = None


class VoicePipeline:
    def __init__(self, *, segmenter: UtteranceSegmenter, stt: SpeechToText,
                 respond: Callable[[str], Awaitable[tuple[str, str]]],
                 tts: TextToSpeech | None = None, wake: WakeWordDetector | None = None,
                 gate: TranscriptQualityGate | None = None, language_hint: str | None = None):
        self.segmenter = segmenter
        self.stt = stt
        self.respond = respond  # async (text) -> (reply_text, reply_language)
        self.tts = tts
        self.wake = wake
        self.gate = gate or TranscriptQualityGate()
        self.language_hint = language_hint
        self.state = State.WAITING_FOR_WAKE if wake else State.LISTENING

    async def feed(self, frame: bytes) -> VoiceTurn | None:
        if self.state == State.WAITING_FOR_WAKE:
            if self.wake and self.wake.detect(frame):
                log.info("voice.wake")
                self.state = State.LISTENING
            return None
        pcm = self.segmenter.push(frame)
        if pcm is None:
            return None
        turn = await self._handle_utterance(pcm)
        if self.wake:
            self.state = State.WAITING_FOR_WAKE
        return turn

    async def _handle_utterance(self, pcm: bytes) -> VoiceTurn:
        tr = self.stt.transcribe(pcm, self.language_hint)
        verdict = self.gate.check(tr)
        if not verdict.accepted:
            log.info("voice.rejected", extra={"reason": verdict.reason})
            return VoiceTurn(transcript=tr.text, accepted=False, reason=verdict.reason)
        reply, lang = await self.respond(tr.text)
        audio = None
        if self.tts:
            try:
                audio = self.tts.synthesize(reply, lang)
            except Exception as e:  # speech output failure must not lose the text reply
                log.warning("voice.tts_failed", extra={"error": str(e)})
        return VoiceTurn(transcript=tr.text, accepted=True, reason="ok", reply_text=reply,
                         reply_audio=audio, language=lang)
