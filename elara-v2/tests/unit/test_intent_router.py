import pytest

from elara.agent.intent import Intent, IntentClassifier, extract_math
from elara.agent.replies import parse_yes_no
from elara.agent.router import Router, Tier
from elara.memory import MemoryPolicy

clf = IntentClassifier(MemoryPolicy())


@pytest.mark.parametrize("text,intent", [
    ("Salam ELARA", Intent.GREETING),
    ("Merhaba", Intent.GREETING),
    ("Salam ELARA, necəsən?", Intent.HOW_ARE_YOU),
    ("how are you?", Intent.HOW_ARE_YOU),
    ("Sağ ol!", Intent.THANKS),
    ("thanks", Intent.THANKS),
    ("What's 17 * 42?", Intent.CALCULATION),
    ("17*42 neçədir?", Intent.CALCULATION),
    ("hesabla 2^10", Intent.CALCULATION),
    ("sqrt(2)", Intent.CALCULATION),
    ("What time is it?", Intent.TIME),
    ("Remember that my favorite programming language is Python.", Intent.MEMORY_STORE),
    ("Forget that I like tea", Intent.MEMORY_FORGET),
    ("What do you know about me?", Intent.MEMORY_LIST),
    ("What's my favorite programming language?", Intent.MEMORY_QUERY),
    ("Find recent research about single-cell RNA sequencing", Intent.RESEARCH),
    ("CRISPR haqqında son məqalələri tap", Intent.RESEARCH),
    ("Is rs80357906 pathogenic?", Intent.RESEARCH),
    ("Open my project folder", Intent.OPEN_PATH),
    ("list files in ~/Documents", Intent.FILE_LIST),
    ("What is PCR?", Intent.GENERAL),
    ("Write a haiku about the Caspian sea", Intent.GENERAL),
    ("What is the difference between 2 and 3 year plans?", Intent.GENERAL),
])
def test_classify(text, intent):
    assert clf.classify(text).intent == intent


def test_reference_only_when_references_exist():
    assert clf.classify("tell me more about the second one").intent == Intent.GENERAL
    r = clf.classify("tell me more about the second one", has_references=True)
    assert r.intent == Intent.REFERENCE and r.slots["index"] == 2
    assert clf.classify("ikinci məqaləni izah et", has_references=True).slots["index"] == 2
    assert clf.classify("that paper — is it a review?", has_references=True).slots["index"] == 0
    assert clf.classify("open #3", has_references=True).slots["index"] == 3


def test_extract_math_rejects_non_math():
    assert extract_math("What's the meaning of life?") is None
    assert extract_math("2024") is None
    assert extract_math("what is 3 + 4") == "3 + 4"


def test_router_tiers():
    r = Router()
    assert r.route(Intent.CALCULATION, "1+1").tier == Tier.DETERMINISTIC
    assert r.route(Intent.RESEARCH, "x").tier == Tier.RESEARCH
    assert r.route(Intent.GENERAL, "hi there, tell me a joke").tier == Tier.FAST
    assert r.route(Intent.GENERAL, "Analyze the trade-offs of SQLite vs Postgres").tier == \
        Tier.STRONG
    assert r.route(Intent.GENERAL, "x" * 500).tier == Tier.STRONG
    assert r.route(Intent.GENERAL, "x").model_tier == "fast"


@pytest.mark.parametrize("text,val", [("yes", True), ("Bəli", True), ("hə", True),
                                      ("evet", True), ("no", False), ("xeyr", False),
                                      ("hayır", False), ("maybe later", None)])
def test_yes_no(text, val):
    assert parse_yes_no(text) is val
