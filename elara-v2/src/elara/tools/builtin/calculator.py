"""Safe arithmetic evaluation (AST whitelist, no eval)."""

from __future__ import annotations

import ast
import math
import operator
from typing import ClassVar

from pydantic import BaseModel, Field

from elara.core.errors import ToolError
from elara.tools.base import Permission, Tool, ToolContext

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
           ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "asin": math.asin, "acos": math.acos, "atan": math.atan, "log": math.log,
          "ln": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp, "abs": abs,
          "round": round, "floor": math.floor, "ceil": math.ceil, "factorial": math.factorial}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
MAX_EXPONENT = 10_000
MAX_FACTORIAL = 1_000


def normalize_expression(expr: str) -> str:
    return (expr.replace("×", "*").replace("·", "*").replace("÷", "/").replace("^", "**")
            .replace("−", "-").replace("√", "sqrt").strip().rstrip("=?").strip())


def evaluate(expr: str) -> float | int:
    expr = normalize_expression(expr)
    if not expr or len(expr) > 500:
        raise ToolError("expression is empty or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"not a valid expression: {expr!r}") from e
    try:
        return _eval(tree.body)
    except ZeroDivisionError as e:
        raise ToolError("division by zero") from e
    except (OverflowError, ValueError) as e:
        raise ToolError(f"math error: {e}") from e


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("exponent too large")
        if isinstance(node.op, ast.Pow) and abs(left) > 1 and abs(right) * math.log10(
                abs(left)) > 4000:
            raise ToolError("result too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and (not args or args[0] > MAX_FACTORIAL):
            raise ToolError("factorial argument too large")
        return _FUNCS[node.func.id](*args)
    raise ToolError(f"unsupported element in expression: {ast.dump(node)[:60]}")


def format_number(v: float | int) -> str:
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"
    return str(v)


class CalculatorInput(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '17 * 42' or 'sqrt(2)/3'")


class CalculatorOutput(BaseModel):
    expression: str
    result: float | int
    formatted: str


class CalculatorTool(Tool):
    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = ("Evaluate an arithmetic expression exactly (+ - * / // % **, "
                                  "sqrt, log, sin, cos, pi, e...). Use for any math.")
    Input = CalculatorInput
    Output = CalculatorOutput
    permission = Permission.READ_ONLY
    category = "math"
    timeout_s = 5.0

    async def run(self, args: CalculatorInput, ctx: ToolContext) -> CalculatorOutput:
        value = evaluate(args.expression)
        return CalculatorOutput(expression=normalize_expression(args.expression), result=value,
                                formatted=format_number(value))
