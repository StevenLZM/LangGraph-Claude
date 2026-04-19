"""SearchTool 协议 + ToolResult。详见 ENGINEERING.md §7.1。"""
from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

SourceType = Literal["web", "academic", "code", "kb"]


class ToolResult(TypedDict, total=False):
    snippet: str
    source_url: str
    relevance_score: float
    extra: dict[str, Any]


@runtime_checkable
class SearchTool(Protocol):
    name: str
    source_type: SourceType

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]: ...

    async def close(self) -> None: ...
