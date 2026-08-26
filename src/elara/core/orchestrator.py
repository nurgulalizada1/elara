"""ELARA orchestrator."""

from __future__ import annotations

from elara.llm.provider import LLMProvider
from elara.memory.persistent import PersistentMemory
from elara.memory.profile import UserProfile
from elara.memory.store import ConversationMemory


class ElaraOrchestrator:
    """Coordinates user input, memory, profile, and LLM responses."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: ConversationMemory | None = None,
        profile: UserProfile | None = None,
        persistent_memory: PersistentMemory | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory or ConversationMemory()
        self.persistent_memory = persistent_memory

        if profile is not None:
            self.profile = profile
        elif self.persistent_memory is not None:
            self.profile = self.persistent_memory.load_profile()
        else:
            self.profile = UserProfile()

    def _save_profile(self) -> None:
        """Persist the profile when persistent memory is enabled."""

        if self.persistent_memory is not None:
            self.persistent_memory.save_profile(self.profile)

    def handle(self, user_text: str) -> str:
        """Handle a user message."""

        if not user_text.strip():
            return "Mesaj boş ola bilməz."

        self.memory.add("user", user_text)

        confirmation_response = self._handle_name_confirmation(user_text)

        if confirmation_response is not None:
            self._save_profile()
            self.memory.add("assistant", confirmation_response)
            return confirmation_response

        local_response = self._handle_local_memory(user_text)

        if local_response is not None:
            self.memory.add("assistant", local_response)
            return local_response

        profile_response = self._update_profile(user_text)

        if profile_response is not None:
            self._save_profile()
            self.memory.add("assistant", profile_response)
            return profile_response

        self._save_profile()

        prompt = self._build_prompt()
        response = self.provider.generate(prompt)

        self.memory.add("assistant", response)

        return response

    def _extract_name(self, user_text: str) -> str | None:
        """Extract a name from 'Mənim adım X'."""

        normalized = user_text.strip()
        prefix = "Mənim adım "

        if not normalized.lower().startswith(prefix.lower()):
            return None

        name = normalized[len(prefix):].strip()
        name = name.rstrip("?!., ")

        if not name:
            return None

        if name.lower() in {"nədir", "nədi"}:
            return None

        return name

    def _display_name(self, name: str) -> str:
        """Convert 'Aylindir' into the conversational form 'Aylin'."""

        if name.lower().endswith("dir") and len(name) > 3:
            return name[:-3]

        return name

    def _update_profile(self, user_text: str) -> str | None:
        """Extract a name and require confirmation before changing it."""

        new_name = self._extract_name(user_text)

        if new_name is None:
            return None

        if self.profile.name is None:
            self.profile.name = new_name
            self.profile.pending_name = None
            return None

        if self.profile.name.lower() == new_name.lower():
            self.profile.pending_name = None
            return None

        self.profile.pending_name = new_name

        old_name = self.profile.name
        display_name = self._display_name(new_name)

        return (
            f"Axı əvvəl adının {old_name} olduğunu demişdin. "
            f"Bunu {display_name} olaraq dəyişək?"
        )

    def _handle_name_confirmation(self, user_text: str) -> str | None:
        """Handle yes/no responses to a pending name change."""

        if self.profile.pending_name is None:
            return None

        normalized = user_text.strip().lower().rstrip("?!., ")

        yes_answers = {
            "bəli",
            "belə",
            "hə",
            "he",
            "yes",
            "ok",
            "oldu",
        }

        no_answers = {
            "xeyr",
            "yox",
            "xeyir",
            "no",
        }

        if normalized in yes_answers:
            new_name = self.profile.pending_name
            self.profile.name = new_name
            self.profile.pending_name = None

            display_name = self._display_name(new_name)

            return (
                f"Oldu. Bundan sonra səni {display_name} "
                "kimi yadda saxlayacağam."
            )

        if normalized in no_answers:
            old_name = self.profile.name
            self.profile.pending_name = None

            return (
                f"Oldu, adını dəyişmirəm. "
                f"Sənin adın {old_name} olaraq qalır."
            )

        return None

    def _handle_local_memory(self, user_text: str) -> str | None:
        """Answer simple factual questions using structured memory."""

        normalized = user_text.strip().lower().rstrip("?!., ")

        name_questions = {
            "mənim adım nədir",
            "adım nədir",
            "mən kiməm",
        }

        if normalized in name_questions and self.profile.name:
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
