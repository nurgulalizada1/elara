from elara.llm.provider import LLMProvider
from elara.memory.store import ConversationMemory


class ElaraOrchestrator:
    """Coordinates user input, memory, and LLM responses."""

    def __init__(
        self,
        provider: LLMProvider,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.provider = provider
        self.memory = memory or ConversationMemory()

    def handle(self, user_text: str) -> str:
        """Handle a user message using memory and the LLM provider."""

        if not user_text.strip():
            return "Mesaj boş ola bilməz."

        self.memory.add("user", user_text)

        local_response = self._handle_local_memory(user_text)

        if local_response is not None:
            self.memory.add("assistant", local_response)
            return local_response

        prompt = self._build_prompt()
        response = self.provider.generate(prompt)

        self.memory.add("assistant", response)

        return response

    def _handle_local_memory(self, user_text: str) -> str | None:
        """Handle simple memory-based questions without an LLM."""

        normalized = user_text.strip().lower()

        if "mənim adım nədir" in normalized:
            for message in reversed(self.memory.get_messages()):
                if message["role"] != "user":
                    continue

                content = message["content"].strip()
                content_normalized = content.lower()

                prefix = "mənim adım "

                if content_normalized.startswith(prefix):
                    name = content[len(prefix):].strip()
                    name = name.rstrip("?!.,")

                    if name and name.lower() != "nədir":
                        return f"Sənin adın {name}."

        return None

    def _build_prompt(self) -> str:
        """Build a prompt containing the conversation history."""

        lines = []

        for message in self.memory.get_messages():
            role = message["role"]
            content = message["content"]

            lines.append(f"{role}: {content}")

        return "\n".join(lines)
