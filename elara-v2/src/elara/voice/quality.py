"""Transcript quality gate: reject empty, low-confidence or hallucination-like transcripts
before they reach the assistant (STT models hallucinate on noise)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from elara.voice.interfaces import Transcript

# Common Whisper hallucinations on silence/noise (multilingual).
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.", "you", "bye.",
    "subtitles by the amara.org community", "altyazı m.k.", "abunə olun", "izlediğiniz için "
    "teşekkürler", "teşekkürler.", "sağ olun.", ".", "...",
}


@dataclass
class GateResult:
    accepted: bool
    reason: str


class TranscriptQualityGate:
    def __init__(self, min_confidence: float = 0.45, max_no_speech: float = 0.6,
                 min_chars: int = 2, min_duration_s: float = 0.3):
        self.min_confidence = min_confidence
        self.max_no_speech = max_no_speech
        self.min_chars = min_chars
        self.min_duration_s = min_duration_s

    def check(self, tr: Transcript) -> GateResult:
        text = tr.text.strip()
        if len(re.sub(r"\W", "", text)) < self.min_chars:
            return GateResult(False, "empty transcript")
        if tr.duration_s and tr.duration_s < self.min_duration_s:
            return GateResult(False, "audio too short")
        if tr.no_speech_prob is not None and tr.no_speech_prob > self.max_no_speech:
            return GateResult(False, f"probably not speech (p={tr.no_speech_prob:.2f})")
        if tr.confidence is not None and tr.confidence < self.min_confidence:
            return GateResult(False, f"low confidence ({tr.confidence:.2f})")
        if text.lower() in HALLUCINATIONS:
            return GateResult(False, "known STT hallucination")
        words = text.lower().split()
        if len(words) >= 6 and len(set(words)) <= len(words) // 4:
            return GateResult(False, "repetitive output")
        return GateResult(True, "ok")
