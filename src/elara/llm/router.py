from elara.llm.provider import LLMProvider


class LLMRouter:
    """Selects an LLM provider for ELARA."""

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        self.providers = providers

    def generate(self, prompt: str, provider_name: str = "mock") -> str:
        """Generate a response using the selected provider."""

        if provider_name not in self.providers:
            raise ValueError(f"Naməlum provider: {provider_name}")

        return self.providers[provider_name].generate(prompt)
