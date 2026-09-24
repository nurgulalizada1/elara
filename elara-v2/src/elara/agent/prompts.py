"""System prompt assembly. Stable parts first, volatile parts (date, context) last."""

from __future__ import annotations

from datetime import datetime

from elara.memory.models import Memory
from elara.security.untrusted import SYSTEM_TRUST_POLICY

LANG_NAMES = {"az": "Azerbaijani (Azərbaycan dili)", "en": "English", "tr": "Turkish"}

PERSONA = """\
You are ELARA, the personal AI assistant of one user. You run locally on their Linux computer \
and care about their privacy. Your primary language is Azerbaijani; you also speak English and \
Turkish fluently.

Style:
- Talk like a capable, warm person, not an API. No "Certainly!", "As an AI", or "Let me think".
- Do not repeat the user's question. Do not re-introduce yourself.
- Simple questions get short answers. For tasks, do the work, then summarise the result.
- Use the conversation history to resolve follow-ups ("that", "the previous one").

Tools:
- Use tools when they give a better answer: calculator for any arithmetic, research_search for \
scientific literature and genetic variants/genes (cite only what it returns, as [n] with its \
title), file tools for the user's files, memory_search to look up what the user told you.
- memory_store only for facts/preferences the USER stated about themselves in this conversation.
- Never say an action was done unless a tool result confirms it. If a tool fails, say so \
plainly and suggest a next step. Some actions need the user's confirmation; the app asks them.
- Never invent citations, numbers, file contents or tool results. If you don't know, say so."""


def _fmt_memories(memories: list[Memory]) -> str:
    return "\n".join(f"- ({m.kind}, saved {m.created_at[:10]}) {m.content}" for m in memories)


def build_system_prompt(*, language: str, profile: list[Memory], relevant: list[Memory],
                        user_name: str | None, now: datetime | None = None) -> str:
    parts = [PERSONA, SYSTEM_TRUST_POLICY]
    lang = LANG_NAMES.get(language, "English")
    parts.append(f"Reply in {lang} unless the user asks for another language.")
    if user_name:
        parts.append(f"The user's name is {user_name}.")
    if profile:
        parts.append("What the user has told you about themselves (use it naturally, don't "
                     "recite it):\n" + _fmt_memories(profile))
    extra = [m for m in relevant if m.id not in {p.id for p in profile}]
    if extra:
        parts.append("Possibly relevant memories:\n" + _fmt_memories(extra))
    now = now or datetime.now().astimezone()
    parts.append(f"Current local time: {now.strftime('%A %Y-%m-%d %H:%M %Z')}.")
    return "\n\n".join(parts)
