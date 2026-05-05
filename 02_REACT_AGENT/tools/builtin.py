"""LangChain tool definitions for the ReAct demo."""
from __future__ import annotations

import ast
import math
import os
import re
import unicodedata
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from sandbox.executor import run_python_code
from config.settings import settings

_GARBLED_CONTENT_FALLBACK = "摘要不可用（搜索结果内容疑似乱码，请打开来源核对）。"


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词，应简洁明确")
    max_results: int = Field(default=3, ge=1, le=5, description="返回结果数量")


def _looks_garbled_search_content(text: str) -> bool:
    sample = text[:300]
    chars = [ch for ch in sample if not ch.isspace()]
    if len(chars) < 30:
        return False
    if "\ufffd" in sample:
        return True

    letters = sum(ch.isalpha() for ch in chars)
    cjk_or_latin = sum("\u4e00" <= ch <= "\u9fff" or "a" <= ch.lower() <= "z" for ch in chars)
    digits = sum(ch.isdigit() for ch in chars)
    punctuation = sum(unicodedata.category(ch).startswith("P") for ch in chars)
    suspicious = sum(unicodedata.category(ch)[0] in {"C", "M", "S"} for ch in chars)
    total = len(chars)

    if suspicious / total > 0.08 and cjk_or_latin / total < 0.35:
        return True
    return (digits + punctuation + suspicious) / total > 0.75 and letters / total < 0.25


def _format_search_content(content: Any, *, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    text = "".join(ch for ch in text if ch.isprintable()).strip()
    if not text:
        return "摘要不可用。"
    if _looks_garbled_search_content(text):
        return _GARBLED_CONTENT_FALLBACK
    return text[:limit]


@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 3) -> str:
    """搜索互联网获取实时信息，如新闻、股价、当前事件。"""
    api_key = os.getenv("TAVILY_API_KEY") or settings.tavily_api_key
    if not api_key:
        return "搜索工具未配置：请设置 TAVILY_API_KEY 后再查询实时互联网信息。"
    try:
        from tavily import TavilyClient
    except Exception:
        return "搜索工具依赖未安装：请安装 tavily-python。"

    try:
        response = TavilyClient(api_key=api_key).search(query=query, max_results=max_results)
    except Exception as exc:
        return f"搜索失败: {exc}"

    results = []
    for item in response.get("results", []):
        title = item.get("title", "无标题")
        content = _format_search_content(item.get("content", ""))
        url = item.get("url", "")
        results.append(f"标题: {title}\n内容: {content}\n来源: {url}")
    return "\n\n---\n\n".join(results) if results else "未搜索到相关结果。"


class CalcInput(BaseModel):
    expression: str = Field(description="数学表达式，如 1234 * 5678 或 sqrt(144)")


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}
_FUNCS: dict[str, Any] = {
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
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        args = [_safe_eval(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError("仅支持数字、四则运算、幂、取模和少量数学函数")


@tool("calculator", args_schema=CalcInput)
def calculator(expression: str) -> str:
    """执行精确数学计算。数学问题必须优先使用此工具。"""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
    except Exception as exc:
        return f"计算错误: {exc}"
    return f"计算结果: {expression} = {result}"


class CodeInput(BaseModel):
    code: str = Field(description="要执行的 Python 代码，必须是完整代码片段")


@tool("python_executor", args_schema=CodeInput)
def python_executor(code: str) -> str:
    """在受限环境中执行 Python 代码。禁止文件系统、网络、导入和动态执行。"""
    return run_python_code(code)


class DatetimeInput(BaseModel):
    timezone: str = Field(default="Asia/Shanghai", description="IANA 时区名")


@tool("get_datetime", args_schema=DatetimeInput)
def get_datetime(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前日期、时间和星期。"""
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return f"时间工具错误: 未识别时区 {timezone}"
    now = datetime.now(tz)
    weekday = ["一", "二", "三", "四", "五", "六", "日"][now.weekday()]
    return (
        f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期: {weekday}\n"
        f"时区: {timezone}"
    )


class WikiInput(BaseModel):
    query: str = Field(description="维基百科词条关键词")


@tool("wikipedia_search", args_schema=WikiInput)
def wikipedia_search(query: str) -> str:
    """查询维基百科概念背景；不适合实时信息。"""
    try:
        import wikipedia
    except Exception:
        return "Wikipedia 工具依赖未安装：请安装 wikipedia。"

    wikipedia.set_lang("zh")
    try:
        page = wikipedia.page(query, auto_suggest=False)
        return page.summary[:1000]
    except Exception as exc:
        return f"Wikipedia 查询失败: {exc}"


def get_builtin_tools():
    return [web_search, calculator, python_executor, get_datetime, wikipedia_search]


def get_tools():
    """Backward-compatible alias for local built-in tools only."""
    return get_builtin_tools()
