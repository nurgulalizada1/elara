from elara.core.orchestrator import ElaraOrchestrator
from elara.llm.mock import MockLLMProvider


def test_orchestrator_uses_provider():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    response = orchestrator.handle("Salam")

    assert response == "ELARA cavab verir: Salam"


def test_orchestrator_rejects_empty_message():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    response = orchestrator.handle("")

    assert response == "Mesaj boş ola bilməz."
def test_orchestrator_keeps_conversation_memory():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Salam")
    orchestrator.handle("Mənim adım Nurguldur")

    messages = orchestrator.memory.get_messages()

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Salam"

    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Mənim adım Nurguldur"
def test_orchestrator_sends_memory_to_provider():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Salam")
    orchestrator.handle("Mənim adım Nurguldur")

    messages = orchestrator.memory.get_messages()

    assert messages[0] == {
        "role": "user",
        "content": "Salam",
    }

    assert messages[2] == {
        "role": "user",
        "content": "Mənim adım Nurguldur",
    }
def test_orchestrator_remembers_user_name():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Mənim adım Nurguldur")

    response = orchestrator.handle("Mənim adım nədir?")

    assert response == "Sənin adın Nurguldur."
def test_orchestrator_stores_name_in_profile():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Mənim adım Nurguldur")

    assert orchestrator.profile.name == "Nurguldur"


def test_orchestrator_uses_profile_for_name_question():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Mənim adım Nurguldur")

    response = orchestrator.handle("Mən kiməm?")

    assert response == "Sənin adın Nurguldur."


def test_orchestrator_updates_name():
    provider = MockLLMProvider()
    orchestrator = ElaraOrchestrator(provider)

    orchestrator.handle("Mənim adım Nurguldur")
    orchestrator.handle("Mənim adım Aylindir")

    assert orchestrator.profile.name == "Aylindir"
