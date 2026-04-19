"""跨 turn 状态重置 —— ENGINEERING.md §5.2。"""
from __future__ import annotations

from typing import Any

PER_TURN_RESET_FIELDS = (
    "revision_count",
    "iteration",
    "next_node",
    "coverage_by_subq",
    "missing_aspects",
    "next_action",
)


def reset_per_turn(state: dict[str, Any], new_query: str) -> dict[str, Any]:
    """新一轮追问前清理易变字段；保留 messages / evidence / plan / final_report。"""
    patch: dict[str, Any] = {"research_query": new_query}
    for k in PER_TURN_RESET_FIELDS:
        patch[k] = 0 if k in {"revision_count", "iteration"} else None
    patch["coverage_by_subq"] = {}
    patch["missing_aspects"] = []
    return patch
