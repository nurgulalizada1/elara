import pytest

from elara.memory.answer import LocalAnswerer, coverage, is_personal_question


@pytest.mark.parametrize("q", [
    "What language do I prefer?", "What is my favorite programming language?", "Who am I?",
    "Which editor do I use?", "Mən nə ilə maraqlanıram?", "Mənim adım nədir?",
    "Benim en sevdiğim renk ne?", "Hangi dili tercih ediyorum?",
])
def test_personal_questions(q):
    assert is_personal_question(q)


@pytest.mark.parametrize("q", [
    "What is PCR?", "How do I install Python?", "Can you tell me my options?",
    "What is the capital of France?", "PCR nədir?", "Explain CRISPR to me",
])
def test_not_personal_questions(q):
    assert not is_personal_question(q)


def test_coverage_handles_suffixes():
    assert coverage("What language do I prefer for answers?",
                    "my permanent test preference is Azerbaijani answers.") >= 0.5
    assert coverage("Mən nə ilə maraqlanıram?", "mən astrobiologiya ilə maraqlanıram") == 1.0
    assert coverage("How do I install Python?", "my favorite language is Python") < 1.0


def test_statements_later_wins_and_threshold():
    a = LocalAnswerer()
    hit = a.from_statements("What is my test name?", ["My test name is A.", "My test name is B."])
    assert hit.content == "My test name is B."
    assert a.from_statements("What is my blood type?", ["My test name is A."]) is None
