"""Transcript quality gate: a conservative, deterministic check before text reaches ElaraCore.

Rejects STT failures, empty/noise-only output, filler sounds, low-probability decodes and
known Whisper hallucinations; accepts ordinary short commands ("stop", "cancel",
"open calendar", "what time is it"). Thresholds for faster-whisper metrics are Whisper's
own defaults (log_prob_threshold=-1.0, no_speech_threshold=0.6,
compression_ratio_threshold=2.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from elara.voice.interfaces import Transcript

# Common Whisper outputs on silence/noise. Rejected only when the decode also looks weak
# (or when the engine gives no signal), so a clearly spoken "thank you" still passes.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thank you for watching.", "you",
    "bye.", "subtitles by the amara.org community", "altyazı m.k.", "abunə olun",
    "izlediğiniz için teşekkürler", "teşekkürler.", "sağ olun.",
}
FILLERS = {"uh", "um", "umm", "hmm", "hm", "mm", "mhm", "ah", "eh", "er", "erm", "oh"}


@dataclass
class GateResult:
    accepted: bool
    reason: str
    code: str = "ok"   # ok | stt_failed | empty | too_short_audio | no_speech | low_logprob |
    #                    low_confidence | repetitive | filler | hallucination | wrong_language


class TranscriptQualityGate:
    def __init__(self, min_confidence: float = 0.45, max_no_speech: float = 0.6,
                 min_logprob: float = -1.0, max_compression_ratio: float = 2.4,
                 min_duration_s: float = 0.3, expected_language: str | None = None):
        self.min_confidence = min_confidence
        self.max_no_speech = max_no_speech
        self.min_logprob = min_logprob
        self.max_compression_ratio = max_compression_ratio
        self.min_duration_s = min_duration_s
        self.expected_language = expected_language

    def check(self, tr: Transcript) -> GateResult:
        if tr.error:
            return GateResult(False, f"transcription failed: {tr.error}", "stt_failed")
        text = tr.text.strip()
        lowered = text.lower()
        words = re.sub(r"[^\w\s]", " ", lowered).split()
        if not words:
            return GateResult(False, "empty transcript", "empty")
        if tr.duration_s and tr.duration_s < self.min_duration_s:
            return GateResult(False, "audio too short", "too_short_audio")
        if (self.expected_language and tr.language
                and tr.language != self.expected_language):
            return GateResult(False, f"language '{tr.language}' is not "
                                     f"'{self.expected_language}'", "wrong_language")
        if tr.no_speech_prob is not None and tr.no_speech_prob > self.max_no_speech and (
                tr.avg_logprob is None or tr.avg_logprob < self.min_logprob):
            return GateResult(False, f"probably not speech (p={tr.no_speech_prob:.2f})",
                              "no_speech")
        if tr.avg_logprob is not None and tr.avg_logprob < self.min_logprob:
            return GateResult(False, f"low decode probability (avg_logprob={tr.avg_logprob})",
                              "low_logprob")
        if tr.confidence is not None and tr.confidence < self.min_confidence:
            return GateResult(False, f"low confidence ({tr.confidence:.2f})", "low_confidence")
        if tr.compression_ratio is not None and tr.compression_ratio > self.max_compression_ratio:
            return GateResult(False, f"repetitive output (compression={tr.compression_ratio})",
                              "repetitive")
        if len(words) >= 6 and len(set(words)) <= len(words) // 4:
            return GateResult(False, "repetitive output", "repetitive")
        if all(w in FILLERS for w in words):
            return GateResult(False, "filler sound, not a request", "filler")
        if lowered in HALLUCINATIONS:
            weak = (tr.no_speech_prob is None and tr.avg_logprob is None) or \
                (tr.no_speech_prob is not None and tr.no_speech_prob >= 0.3) or \
                (tr.avg_logprob is not None and tr.avg_logprob < -0.7)
            if weak:
                return GateResult(False, "known STT hallucination", "hallucination")
        return GateResult(True, "ok", "ok")
