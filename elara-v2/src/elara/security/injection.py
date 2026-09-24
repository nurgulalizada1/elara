"""Prompt-injection detection for untrusted content.

This is a *signal*, not the defence. The defence is architectural (see
docs/SECURITY.md): untrusted content is enveloped, never merged into instructions,
taints the turn (which escalates permissions), and can never write user memory.
Detection adds: security-event logging, a visible warning inside the envelope, and
hard blocks where the risk is concentrated (memory writes, tool arguments).

Signals combine multilingual (en/az/tr) pattern families with structural cues:
role/chat-template markers, attempts to close our envelope, invisible characters,
and encoded payloads.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿‪-‮⁦-⁩]")


@dataclass(frozen=True)
class Rule:
    category: str
    weight: float
    pattern: re.Pattern[str]


def _r(category: str, weight: float, pattern: str) -> Rule:
    return Rule(category, weight, re.compile(pattern, re.I | re.M))


RULES: list[Rule] = [
    # Instruction override
    _r("override", 0.6, r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(previous|prior|above|earlier|all|any|your|system|original)\b.{0,30}\b(instructions?|prompts?|rules|directives|guidelines|context)\b"),
    _r("override", 0.6, r"\b(əvvəlki|yuxarıdakı|bütün)\b.{0,30}\b(təlimat|göstəriş|qayda)\w*.{0,30}\b(unut|nəzərə alma|ignor|yox say)\w*"),
    _r("override", 0.6, r"\b(önceki|yukarıdaki|tüm)\b.{0,30}\b(talimat|komut|kural)\w*.{0,30}\b(unut|yoksay|görmezden)\w*"),
    # System-prompt exfiltration
    _r("prompt_leak", 0.5, r"\b(reveal|show|print|repeat|output|display|tell me|leak|dump)\b.{0,30}\b(system|hidden|initial|secret|developer)\s+(prompt|instructions?|message)"),
    _r("prompt_leak", 0.5, r"\bsistem\s+(promptunu|təlimatlarını|istemini|talimatlarını)\b"),
    # Secret exfiltration
    _r("secret_leak", 0.6, r"\b(send|reveal|share|post|email|upload|give|print|show|tell)\b.{0,40}\b(api[\s_-]?keys?|passwords?|secrets?|tokens?|credentials|private keys?|ssh keys?|\.env)\b"),
    _r("secret_leak", 0.5, r"\b(api açar|şifrə|parol|gizli açar|şifre|parola)\w*.{0,30}\b(göndər|göstər|paylaş|gönder|göster|paylaş)\w*"),
    # Command execution
    _r("command", 0.5, r"\b(run|execute|eval)\b.{0,20}\b(this|the following|these|below)\b.{0,20}\b(commands?|code|scripts?|shell)\b"),
    _r("command", 0.7, r"(rm\s+-rf\s+[/~]|curl\s+[^|\n]{0,200}\|\s*(ba|z)?sh|wget\s+[^|\n]{0,200}\|\s*sh|sudo\s+\w+|chmod\s+\+x|mkfs\.|:\(\)\s*\{\s*:\|:&\s*\};:)"),
    _r("command", 0.5, r"\b(bu|aşağıdakı|aşağıdaki)\b.{0,20}\b(əmri|komutu|kodu)\b.{0,20}\b(icra et|çalışdır|çalıştır)\w*"),
    # Data exfiltration
    _r("exfiltration", 0.6, r"\b(send|forward|transmit|post|upload|email)\b.{0,40}\b(this|all|the|user'?s?|conversation|chat|memory|memories|data|information|history)\b.{0,40}\b(to|at)\b\s+\S+"),
    _r("exfiltration", 0.4, r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*=\s*[{$]"),
    # Rule / identity change
    _r("rule_change", 0.5, r"\b(change|update|modify|override|replace)\b.{0,20}\byour\b.{0,20}\b(rules|instructions|guidelines|behaviou?r|persona|identity)\b"),
    _r("rule_change", 0.5, r"\b(you are now|from now on,? you (will|must|are)|new instructions\s*:|act as (an? )?(unrestricted|jailbroken|dan)\b|developer mode|jailbreak)"),
    _r("rule_change", 0.45, r"\b(artıq sən|bundan sonra sən|bundan sonra sen|artık sen)\b"),
    # Impersonation of system / developer channels and chat templates
    _r("impersonation", 0.5, r"^\s*(system|developer|assistant|admin(istrator)?)\s*(message|prompt|note)?\s*:"),
    _r("impersonation", 0.6, r"(<\|im_start\|>|<\|im_end\|>|\[/?INST\]|<<SYS>>|<\|system\|>|<\|begin_of_text\|>|###\s*(system|instruction)s?\b|</?system>|BEGIN SYSTEM PROMPT)"),
    _r("impersonation", 0.5, r"\b(message|instructions?|note|update)\s+from\s+(the\s+)?(developer|administrator|system|anthropic|openai|elara'?s? (creator|developer))\b"),
    # Memory poisoning
    _r("memory_poisoning", 0.6, r"\b(remember|memorize|store|save|note)\b.{0,30}\b(that\s+)?the\s+user\s+(wants|prefers|is|has|asked|likes|needs|authorized)\b"),
    _r("memory_poisoning", 0.5, r"\b(save|store|write|add)\b.{0,20}\b(this|it|following)\b.{0,20}\b(to|in|into)\s+(your\s+)?(memory|long-term memory)\b"),
    _r("memory_poisoning", 0.5, r"\b(istifadəçi|kullanıcı)\b.{0,40}\b(yadda saxla|hatırla|kaydet)\w*"),
    # Envelope escape attempts
    _r("envelope_escape", 0.9, r"</?\s*untrusted_content\b"),
]

_BASE64_BLOB = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")


@dataclass
class InjectionReport:
    score: float = 0.0
    categories: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return self.score >= 0.5

    @property
    def severity(self) -> str:
        if self.score >= 1.0:
            return "high"
        if self.score >= 0.5:
            return "medium"
        return "low" if self.score > 0 else "none"

    def summary(self) -> str:
        return ", ".join(sorted(set(self.categories))) or "none"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"[ \t]+", " ", text)


class InjectionDetector:
    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules or RULES

    def scan(self, text: str) -> InjectionReport:
        report = InjectionReport()
        if not text:
            return report
        if _ZERO_WIDTH.search(text):
            report.score += 0.2
            report.categories.append("hidden_characters")
            report.evidence.append("invisible/bidi control characters")
        norm = normalize(text)
        seen: set[str] = set()
        for rule in self.rules:
            m = rule.pattern.search(norm)
            if not m:
                continue
            # Repeated hits within one category add less than the first.
            report.score += rule.weight if rule.category not in seen else rule.weight * 0.3
            seen.add(rule.category)
            report.categories.append(rule.category)
            report.evidence.append(m.group(0)[:120])
        if _BASE64_BLOB.search(norm) and seen:
            report.score += 0.2
            report.categories.append("encoded_payload")
        report.score = round(min(report.score, 3.0), 2)
        return report
