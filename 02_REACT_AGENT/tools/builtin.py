"""LangChain tool definitions for the ReAct demo."""
from __future__ import annotations

import ast
import asyncio
import math
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from mcp_servers.weather_data import format_weather, get_weather
from sandbox.executor import run_python_code


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词，应简洁明确")
    max_results: int = Field(default=3, ge=1, le=5, description="返回结果数量")


@tool("web_search", args_schema=SearchInput)
def web_search(query: str, max_results: int = 3) -> str:
    """搜索互联网获取实时信息，如新闻、股价、当前事件。"""
    api_key = os.getenv("TAVILY_API_KEY")
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
        content = str(item.get("content", ""))[:300]
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


class WeatherInput(BaseModel):
    city: str = Field(description="城市名称，支持中文，如北京、上海")
    units: str = Field(default="metric", description="metric=摄氏度，imperial=华氏度")


@tool("weather_query", args_schema=WeatherInput)
def weather_query(city: str, units: str = "metric") -> str:
    """通过内部天气 MCP 实现查询天气和户外活动建议。"""
    data = asyncio.run(get_weather(city, units))
    return format_weather(data)


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


def get_tools():
    return [web_search, calculator, python_executor, weather_query, get_datetime, wikipedia_search]
