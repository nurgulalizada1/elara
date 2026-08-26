class ConversationMemory:
    """Stores messages from the current conversation."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """Add a message to memory."""
        self.messages.append({
            "role": role,
            "content": content,
        })

    def get_messages(self) -> list[dict[str, str]]:
        """Return all stored messages."""
        return list(self.messages)

    def clear(self) -> None:
        """Clear the conversation memory."""
        self.messages.clear()
