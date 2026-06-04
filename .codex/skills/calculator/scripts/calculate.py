#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import math
from collections.abc import Callable


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}

_FUNCS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "round": round,
    "sqrt": math.sqrt,
}


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_safe_eval(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError("仅支持数字、四则运算、幂、取模和少量数学函数")


def calculate(expression: str) -> int | float:
    parsed = ast.parse(expression, mode="eval")
    return _safe_eval(parsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely evaluate a numeric expression.")
    parser.add_argument("expression", help="Numeric expression, such as '1234 * 5678' or 'sqrt(144)'")
    args = parser.parse_args()

    try:
        result = calculate(args.expression)
    except Exception as exc:
        print(f"非法计算: {exc}")
        return 1

    print(f"计算结果: {args.expression} = {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
