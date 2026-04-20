"""路由函数 —— ENGINEERING.md §3.3 真实实现。

supervisor_route: Supervisor 出口条件路由 —— 返回 list[Send] 做 fan-out，或 str 指向单节点。
reflector_route:  Reflector 出口条件路由 —— "supervisor" 触发补查，"writer" 收敛。
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Send

from graph.state import ResearchState

_SOURCE_TO_NODE = {
    "web": "web_researcher",
    "academic": "academic_researcher",
    "code": "code_researcher",
    "kb": "kb_researcher",
}


def supervisor_route(state: ResearchState) -> Any:
    # 计划未确认 → 回到 planner（HITL 恢复后再跑）
    if not state.get("plan_confirmed"):
        return "planner"

    # 硬兜底：revision_count 超限直接出报告
    if state.get("revision_count", 0) >= 3:
        return "writer"

    # 已经有plan但还没收集到evidence(首轮)
    # or 
    # 反思后触发补查 
    # → fan-out 派发
    plan = state.get("plan") or []
    if (plan and not state.get("evidence")) or (state.get("next_action") == "need_more_research"):
        sends: list[Send] = []
        for sq in plan:
            print(f"子问题：{sq}")
            # 取 sq.status 如果没有 status 属性，默认 "pending"
            if getattr(sq, "status", "pending") == "done":
                continue
            sources = getattr(sq, "recommended_sources", []) or ["web"]
            for src in sources:
                node = _SOURCE_TO_NODE.get(src)
                if not node:
                    continue
                sends.append(
                    Send(
                        node,
                        {
                            "sub_question": sq,
                            "research_query": state.get("research_query", ""),
                        },
                    )
                )
        return sends or "writer"

    return "writer"


def reflector_route(state: ResearchState) -> str:
    if state.get("revision_count", 0) >= 3:
        return "writer"
    action = state.get("next_action", "sufficient")
    return "supervisor" if action == "need_more_research" else "writer"
