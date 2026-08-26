"""ELARA orchestrator.

Coordinates conversation memory, structured user memory,
and the LLM provider.
"""

from __future__ import annotations

from elara.llm.provider import LLMProvider
from elara.memory.profile import UserProfile
from elara.memory.store import ConversationMemory


class ElaraOrchestrator:
    """Coordinates user input, memory, and LLM responses."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: ConversationMemory | None = None,
        profile: UserProfile | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory or ConversationMemory()
        self.profile = profile or UserProfile()

        self.profile.pending_name: str | None = None

    def handle(self, user_text: str) -> str:
        """Handle a user message."""

        if not user_text.strip():
            return "Mesaj boş ola bilməz."

        self.memory.add("user", user_text)

        # First handle pending confirmation.
        confirmation_response = self._handle_confirmation(user_text)

        if confirmation_response is not None:
            self.memory.add("assistant", confirmation_response)
            return confirmation_response

        # Handle simple factual questions using structured memory.
        local_response = self._handle_local_memory(user_text)

        if local_response is not None:
            self.memory.add("assistant", local_response)
            return local_response

        # Extract structured facts from normal statements.
        name_response = self._update_profile(user_text)

        if name_response is not None:
            self.memory.add("assistant", name_response)
            return name_response

        # Otherwise use the LLM provider.
        prompt = self._build_prompt()
        response = self.provider.generate(prompt)

        self.memory.add("assistant", response)

        return response

    def _update_profile(self, user_text: str) -> str | None:
        """Extract and possibly update the user's name."""

        normalized = user_text.strip()

        prefix = "Mənim adım "

        if not normalized.lower().startswith(prefix.lower()):
            return None

        name = normalized[len(prefix):].strip()

        # Remove only punctuation, NOT letters from the name.
        name = name.rstrip("?!., ")

        # Do not treat a question as a name.
        if name.lower() in {"nədir", "nədi"}:
            return None

        if not name:
            return None

        # No previous name: save immediately.
        if self.profile.name is None:
            self.profile.name = name
            return None

        # Same name: nothing to change.
        if self.profile.name.casefold() == name.casefold():
            return None

        # Different name: ask for confirmation.
        self.profile.pending_name = name

        return (
            f"Axı əvvəl adının {self.profile.name} olduğunu demişdin. "
            f"Bunu {self._display_name_without_dir(name)} olaraq dəyişək?"
        )

    def _display_name_without_dir(self, name: str) -> str:
        """Return a natural form for confirmation text."""

        if name.lower().endswith("dir") and len(name) > 3:
            return name[:-3]

        return name

    def _handle_confirmation(self, user_text: str) -> str | None:
        """Handle confirmation or rejection of a pending name change."""

        if self.profile.pending_name is None:
            return None

        normalized = user_text.strip().lower().rstrip("?!., ")

        yes_answers = {
            "bəli",
            "belə",
            "hə",
            "hə, dəyiş",
            "dəyiş",
            "bəli, dəyiş",
        }

        no_answers = {
            "xeyr",
            "yox",
            "xeyr, dəyişmə",
            "dəyişmə",
        }

        if normalized in yes_answers:
            new_name = self.profile.pending_name
            self.profile.name = new_name
            self.profile.pending_name = None

            display_name = self._display_name_without_dir(new_name)

            return (
                f"Oldu. Bundan sonra səni {display_name} "
                f"kimi yadda saxlayacağam."
            )

        if normalized in no_answers:
            old_name = self.profile.name
            self.profile.pending_name = None

            return (
                f"Oldu, adını dəyişmirəm. "
                f"Sənin adın {old_name} olaraq qalır."
            )

        # If the user says something unrelated, keep the pending change.
        return None

    def _handle_local_memory(self, user_text: str) -> str | None:
        """Answer simple factual questions using structured memory."""

        normalized = user_text.strip().lower().rstrip("?!., ")

        name_questions = {
            "mənim adım nədir",
            "adım nədir",
            "mən kiməm",
        }

        if normalized in name_questions:
            if self.profile.name:
                return f"Sənin adın {self.profile.name}."

        return None

    def _build_prompt(self) -> str:
        """Build a prompt containing conversation history."""

        lines = []

        for message in self.memory.get_messages():
            role = message["role"]
            content = message["content"]

            lines.append(f"{role}: {content}")

        return "\n".join(lines)
