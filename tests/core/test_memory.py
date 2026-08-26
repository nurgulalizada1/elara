from elara.memory.store import ConversationMemory


def test_memory_stores_messages():
    memory = ConversationMemory()

    memory.add("user", "Salam")
    memory.add("assistant", "Salam! Mən ELARA.")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Salam"
    assert messages[1]["role"] == "assistant"


def test_memory_can_clear():
    memory = ConversationMemory()

    memory.add("user", "Salam")
    memory.clear()

    assert memory.get_messages() == []
