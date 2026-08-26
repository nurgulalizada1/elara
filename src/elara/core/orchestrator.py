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
        """Handle a user message and return the LLM response."""

        if not user_text.strip():
            return "Mesaj boş ola bilməz."

        self.memory.add("user", user_text)

        prompt = self._build_prompt()

        response = self.provider.generate(prompt)

        self.memory.add("assistant", response)

        return response

    def _build_prompt(self) -> str:
        """Build a prompt containing the conversation history."""

        lines = []

        for message in self.memory.get_messages():
            role = message["role"]
            content = message["content"]

            lines.append(f"{role}: {content}")

        return "\n".join(lines)
