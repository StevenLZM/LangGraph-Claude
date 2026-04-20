"""主图装配 —— ENGINEERING.md §3.1。

Python 3.11+ 异步上下文 contextvars 修复后，所有节点恢复纯 async（含 interrupt 触发）。
FastAPI 直接 `await graph.ainvoke()`，事件循环天然不阻塞，HTTP/LLM 并发由 asyncio 调度。
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from agents.planner import planner_node
from agents.reflector import reflector_node
from agents.researcher_academic import academic_researcher_node
from agents.researcher_code import code_researcher_node
from agents.researcher_kb import kb_researcher_node
from agents.researcher_web import web_researcher_node
from agents.supervisor import supervisor_node
from agents.writer import writer_node
from graph.router import reflector_route, supervisor_route
from graph.state import ResearchState


def build_graph(checkpointer: Optional[object] = None):
    # ResearchState全局状态，所有节点共享/膝修改
    wf = StateGraph(ResearchState)

    wf.add_node("planner", planner_node) # 任务拆解
    wf.add_node("supervisor", supervisor_node) # 决策下一步
    wf.add_node("web_researcher", web_researcher_node)
    wf.add_node("academic_researcher", academic_researcher_node)
    wf.add_node("code_researcher", code_researcher_node)
    wf.add_node("kb_researcher", kb_researcher_node)
    wf.add_node("reflector", reflector_node) # 质量评估
    wf.add_node("writer", writer_node) # 输出结果

    # 固定流程
    # 用户输入 → planner → supervisor
    wf.add_edge(START, "planner")
    wf.add_edge("planner", "supervisor")

    wf.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "planner": "planner",
            "writer": "writer",
            "web_researcher": "web_researcher",
            "academic_researcher": "academic_researcher",
            "code_researcher": "code_researcher",
            "kb_researcher": "kb_researcher",
        },
    )

    for r in ("web_researcher", "academic_researcher", "code_researcher", "kb_researcher"):
        wf.add_edge(r, "reflector")

    wf.add_conditional_edges(
        "reflector",
        reflector_route,
        {"supervisor": "supervisor", "writer": "writer"},
    )

    wf.add_edge("writer", END)

    # 变成可执行 Agent
    # checkpointer = 状态持久化（断点恢复 / HITL 必备）
    return wf.compile(checkpointer=checkpointer)
