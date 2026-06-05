"""路由函数 —— ENGINEERING.md §3.3 真实实现。

supervisor_route: Supervisor 出口条件路由 —— 返回 str 指向下一阶段。
reflector_route:  Reflector 出口条件路由 —— "supervisor" 触发补查，"writer" 收敛。
"""
from __future__ import annotations

from graph.research_subgraph import build_research_sends
from graph.state import ResearchState


def supervisor_route(state: ResearchState) -> str:
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
        return "research_subgraph" if build_research_sends(state) else "writer"

    return "writer"


def reflector_route(state: ResearchState) -> str:
    if state.get("revision_count", 0) >= 3:
        return "writer"
    action = state.get("next_action", "sufficient")
    return "supervisor" if action == "need_more_research" else "writer"
