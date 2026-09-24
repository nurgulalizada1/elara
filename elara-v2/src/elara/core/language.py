"""Lightweight language identification for Azerbaijani, Turkish and English.

Deterministic and dependency-free: letter evidence (ə is Azerbaijani-only; x/q are
rare in Turkish; ğ/ı/ş/ç/ö/ü rule out English) plus small stopword sets. Short or
ambiguous input falls back to the conversation's previous language or the default.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-zA-ZəƏçÇğĞıIİöÖşŞüÜ']+")

_AZ = {
    "mən", "sən", "biz", "siz", "nə", "necə", "niyə", "harada", "hansı", "və", "ilə", "üçün",
    "deyil", "var", "yox", "salam", "necəsən", "çox", "sağ", "ol", "xahiş", "edirəm", "bəli",
    "xeyr", "hə", "bu", "o", "da", "də", "ki", "mənim", "sənin", "bir", "edə", "bilərsən",
    "yadda", "saxla", "unut", "haqqında", "tap", "aç", "göstər", "nədir", "neçə", "neçədir",
    "indi", "sabah", "bugün", "bu gün", "istəyirəm", "lazımdır", "olar", "axtar", "məqalə",
    "tədqiqat", "gen", "həm", "amma", "çünki", "əgər", "hansı", "kimi", "mənə", "sənə",
    "eləmə", "elə", "et", "edir", "hesabla", "təşəkkür", "sağol", "gecən", "sabahın", "xeyir",
}
_TR = {
    "ben", "sen", "biz", "siz", "ne", "nasıl", "neden", "nerede", "hangi", "ve", "ile", "için",
    "değil", "var", "yok", "merhaba", "nasılsın", "çok", "teşekkür", "teşekkürler", "ederim",
    "lütfen", "evet", "hayır", "bu", "şu", "o", "da", "de", "ki", "benim", "senin", "bir",
    "misin", "mısın", "musun", "mi", "mı", "mu", "mü", "hatırla", "unut", "hakkında", "bul",
    "aç", "göster", "nedir", "kaç", "şimdi", "yarın", "bugün", "istiyorum", "gerek", "ara",
    "makale", "araştırma", "ama", "çünkü", "eğer", "gibi", "bana", "sana", "yap", "hesapla",
    "selam", "günaydın", "iyi", "kim", "neler", "şey", "olarak", "daha", "en",
}
_EN = {
    "i", "you", "we", "they", "what", "how", "why", "where", "which", "and", "with", "for",
    "not", "is", "are", "am", "hello", "hi", "hey", "thanks", "thank", "please", "yes", "no",
    "this", "that", "the", "a", "an", "of", "to", "in", "my", "your", "can", "could", "would",
    "remember", "forget", "about", "find", "open", "show", "what's", "now", "tomorrow",
    "today", "want", "need", "search", "paper", "research", "but", "because", "if", "like",
    "me", "do", "does", "it", "on", "be", "tell", "recent", "list", "files", "folder",
}

SUPPORTED = ("az", "en", "tr")


def detect_language(text: str, fallback: str = "az") -> str:
    lowered = text[:1000].lower()
    words = _WORD.findall(lowered)
    if not words:
        return fallback
    scores = {"az": 0.0, "tr": 0.0, "en": 0.0}
    for w in words:
        scores["az"] += w in _AZ
        scores["tr"] += w in _TR
        scores["en"] += w in _EN
    # Letter evidence.
    if "ə" in lowered:
        scores["az"] += 3 + lowered.count("ə")
    if re.search(r"[ğış]", lowered):
        scores["en"] -= 2
    if re.search(r"[çöü]", lowered):
        scores["en"] -= 1
    # Turkish orthography rarely uses x/q/w inside native words; Azerbaijani does (x, q).
    native = [w for w in words if not w.isascii()]
    if any(("x" in w or "q" in w) for w in native):
        scores["az"] += 1
    # Turkish-only markers.
    if re.search(r"\B(ıyor|iyor|uyor|üyor)", lowered):
        scores["tr"] += 2
    if re.search(r"\B(ıram|irəm|uram|ürəm)\b", lowered):
        scores["az"] += 2

    best = max(scores, key=lambda k: scores[k])
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] <= 0 or ordered[0] == ordered[1]:
        # Tie-break: prefer fallback if it is among the leaders.
        leaders = [k for k, v in scores.items() if v == ordered[0]]
        if fallback in leaders or ordered[0] <= 0:
            return fallback
        return leaders[0]
    return best
