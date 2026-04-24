"""SearchTool 协议 + ToolResult。详见 ENGINEERING.md §7.1。"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

SourceType = Literal["web", "academic", "code", "kb"]


class ToolResult(TypedDict, total=False):
    snippet: str
    source_url: str
    relevance_score: float
    extra: dict[str, Any]


"""
TavilyTool 被认为是 SearchTool 的真实原因：

    主要原因（90%）：TavilyTool 实现了 SearchTool Protocol 要求的所有属性和方法

        有 name 属性

        有 source_type 属性

        有 search 方法

        有 close 方法

    辅助原因（10%）：@runtime_checkable 装饰器允许运行时检查 isinstance(tavilyTool, SearchTool)
"""
@runtime_checkable
class SearchTool(Protocol):
    name: str
    source_type: SourceType

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]: ...

    async def close(self) -> None: ...
