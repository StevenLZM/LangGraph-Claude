"""Small restricted Python executor for demo use."""
from __future__ import annotations

import contextlib
import io
from typing import Any

FORBIDDEN_SNIPPETS = (
    "import ",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "input(",
    "subprocess",
    "socket",
    "requests",
    "pathlib",
    "shutil",
)

SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def run_python_code(code: str) -> str:
    for forbidden in FORBIDDEN_SNIPPETS:
        if forbidden in code:
            return f"安全限制: 禁止使用 '{forbidden}'"

    stdout = io.StringIO()
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, globals_dict, {})
    except Exception as exc:
        return f"执行错误:\n{type(exc).__name__}: {exc}"

    output = stdout.getvalue().strip()
    return f"执行成功:\n{output or '（无输出）'}"
