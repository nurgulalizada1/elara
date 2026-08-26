from elara.llm.mock import MockLLMProvider


def test_mock_provider_generates_response():
    provider = MockLLMProvider()

    response = provider.generate("Salam")

    assert response == "ELARA cavab verir: Salam"
