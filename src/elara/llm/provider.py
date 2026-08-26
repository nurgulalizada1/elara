from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for all ELARA language model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        raise NotImplementedError
