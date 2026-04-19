"""ToolRegistry —— 按 source_type 管理工具降级链。详见 ENGINEERING.md §7.2。"""
from __future__ import annotations

from collections import defaultdict

from tools.base import SearchTool, SourceType


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, list[SearchTool]] = defaultdict(list)

    def register(self, tool: SearchTool) -> None:
        self._tools[tool.source_type].append(tool)

    def get_chain(self, source_type: SourceType) -> list[SearchTool]:
        """返回该源的降级链：主工具 → 兜底 → ..."""
        return list(self._tools.get(source_type, []))

    async def close_all(self) -> None:
        for chain in self._tools.values():
            for tool in chain:
                try:
                    await tool.close()
                except Exception:
                    pass

    def __repr__(self) -> str:
        summary = {k: [t.name for t in v] for k, v in self._tools.items()}
        return f"ToolRegistry({summary})"
