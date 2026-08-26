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
