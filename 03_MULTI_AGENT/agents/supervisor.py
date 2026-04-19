"""Supervisor —— async（无 I/O，但保持统一签名）。"""
from __future__ import annotations

from typing import Any

from graph.state import ResearchState


async def supervisor_node(state: ResearchState) -> dict[str, Any]:
    return {
        "current_node": "supervisor",
        "iteration": state.get("iteration", 0) + 1,
    }
