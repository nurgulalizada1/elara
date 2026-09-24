import pytest

from elara.core.errors import ToolError
from elara.tools.builtin.calculator import evaluate, format_number


@pytest.mark.parametrize("expr,value", [
    ("17 * 42", 714), ("2^10", 1024), ("(1+2)*3", 9), ("10 / 4", 2.5), ("17 × 42", 714),
    ("sqrt(16)", 4.0), ("-3 + 5", 2), ("7 // 2", 3), ("2**0.5", 2 ** 0.5), ("factorial(5)", 120),
])
def test_evaluate(expr, value):
    assert evaluate(expr) == pytest.approx(value)


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "9**9**9", "1/0",
                                  "factorial(10000)", "a + 1", "[1,2]", "True + 1", ""])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        evaluate(expr)


def test_format():
    assert format_number(714.0) == "714"
    assert format_number(1 / 3) == "0.3333333333"
