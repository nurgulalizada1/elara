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

    def handle(self, user_text: str) -> str:
        """Handle a user message."""

        if not user_text.strip():
            return "Mesaj boş ola bilməz."

        self.memory.add("user", user_text)

        # First check whether this is a simple memory question.
        # We do this BEFORE extracting profile data so that
        # "Mənim adım nədir?" cannot overwrite the stored name.
        local_response = self._handle_local_memory(user_text)

        if local_response is not None:
            self.memory.add("assistant", local_response)
            return local_response

        # Extract structured facts from normal user statements.
        self._update_profile(user_text)

        # Build conversation prompt and ask the provider.
        prompt = self._build_prompt()
        response = self.provider.generate(prompt)

        self.memory.add("assistant", response)

        return response

    def _update_profile(self, user_text: str) -> None:
        """Extract simple structured facts from the user's message."""

        normalized = user_text.strip()

        prefix = "Mənim adım "

        if normalized.lower().startswith(prefix.lower()):
            name = normalized[len(prefix):].strip()
            name = name.rstrip("?!., ")

            # Do not save question words as a name.
            if name and name.lower() not in {
                "nədir",
                "nədir?",
                "nədi",
            }:
                self.profile.name = name

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
