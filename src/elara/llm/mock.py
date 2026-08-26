from elara.llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Fake LLM provider used for development and testing."""

    def generate(self, prompt: str) -> str:
        """Return a deterministic response using the latest user message."""

        lines = [
            line
            for line in prompt.splitlines()
            if line.startswith("user: ")
        ]

        if lines:
            latest_message = lines[-1][6:]
        else:
            latest_message = prompt

        return f"ELARA cavab verir: {latest_message}"
