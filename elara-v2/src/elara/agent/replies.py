"""Formatting helpers for deterministic replies."""

from __future__ import annotations

import re

from elara.conversation.i18n import t
from elara.tools.confirmations import PendingAction
from elara.tools.executor import Status, ToolOutcome

_YES = re.compile(r"^(yes|y|yep|yeah|sure|ok|okay|confirm|do it|go ahead|bəli|hə|həə|he|olar|"
                  r"təsdiq(lə|ləyirəm)?|razıyam|evet|tamam|onayla(ıyorum)?|olur)[\s!.]*$", re.I)
_NO = re.compile(r"^(no|n|nope|cancel|stop|don'?t|xeyr|yox|ləğv( et)?|lazım deyil|istəmirəm|"
                 r"hayır|iptal( et)?|vazgeç|gerek yok)[\s!.]*$", re.I)


def parse_yes_no(text: str) -> bool | None:
    s = text.strip().lower()
    if _YES.match(s):
        return True
    if _NO.match(s):
        return False
    return None


def confirmation_prompt(pending: PendingAction, lang: str) -> str:
    return t("confirm_request", lang, action=pending.description, reason=pending.reason)


def outcome_text(outcome: ToolOutcome, lang: str) -> str:
    """User-facing text for a tool run by a deterministic path."""
    if outcome.status == Status.OK:
        out = outcome.output or {}
        done = {"az": "Hazırdır", "en": "Done", "tr": "Tamam"}.get(lang, "Done")
        if outcome.tool == "write_file":
            return f"{done}: {out.get('path')} ({out.get('bytes_written')} B, {out.get('mode')})"
        if outcome.tool == "delete_file":
            return f"{done}: {out.get('path')} ✗"
        if outcome.tool == "open_path":
            return f"{done}: {out.get('path')}"
        if outcome.tool == "run_python":
            parts = [f"exit={out.get('exit_code')}" + (" (timed out)" if out.get("timed_out")
                                                        else "")]
            if out.get("stdout"):
                parts.append(out["stdout"].rstrip())
            if out.get("stderr"):
                parts.append("stderr:\n" + out["stderr"].rstrip())
            return "\n".join(parts)
        return outcome.display or f"{done}."
    if outcome.status == Status.DENIED:
        return t("tool_denied", lang, tool=outcome.tool, error=outcome.error)
    return t("tool_failed", lang, tool=outcome.tool, error=outcome.error)
