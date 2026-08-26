from elara.llm.mock import MockLLMProvider
from elara.llm.router import LLMRouter


def test_router_uses_selected_provider():
    provider = MockLLMProvider()

    router = LLMRouter({
        "mock": provider,
    })

    response = router.generate("Salam", "mock")

    assert response == "ELARA cavab verir: Salam"


def test_router_rejects_unknown_provider():
    router = LLMRouter({
        "mock": MockLLMProvider(),
    })

    try:
        router.generate("Salam", "claude")
    except ValueError as error:
        assert str(error) == "Naməlum provider: claude"
    else:
        raise AssertionError("ValueError gözlənilirdi")
