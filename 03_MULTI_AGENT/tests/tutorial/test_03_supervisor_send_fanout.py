"""配套 TUTORIAL.md 第 5 章 —— Send + Supervisor 路由

学什么：Supervisor 把 research 阶段封装为 research_subgraph，子图内部继续用 Send 并行扇出：
  · "planner" —— plan 未确认
  · "writer"  —— revision_count 超限 / 路径无 evidence 待补
  · "research_subgraph" —— 进入研究子图，由子图 dispatcher 生成 list[Send]

断言：
1. plan_confirmed=False → 回到 planner
2. revision_count >= 3 → 直接 writer
3. plan 已确认 + 无 evidence → 进入 research_subgraph
4. 子图 fan-out 按 sub_question × recommended_sources 笛卡尔生成 Send
5. Send 节点名映射正确（web→web_researcher 等）
6. payload 包含 sub_question 与 research_query
"""
from __future__ import annotations

from langgraph.types import Send

from agents.schemas import SubQuestion
from graph.research_subgraph import build_research_sends
from graph.router import supervisor_route


def test_plan_not_confirmed_returns_planner():
    out = supervisor_route({"plan_confirmed": False})
    assert out == "planner"


def test_revision_overflow_goes_to_writer():
    out = supervisor_route({"plan_confirmed": True, "revision_count": 3})
    assert out == "writer"


def test_first_round_routes_to_research_subgraph():
    plan = [SubQuestion(id="sq1", question="q1", recommended_sources=["web"])]
    out = supervisor_route({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [],
        "research_query": "main query",
    })
    assert out == "research_subgraph"


def test_research_subgraph_dispatcher_builds_send_list():
    plan = [
        SubQuestion(id="sq1", question="q1", recommended_sources=["web", "academic"]),
        SubQuestion(id="sq2", question="q2", recommended_sources=["code"]),
        SubQuestion(id="sq3", question="q3", recommended_sources=["kb"]),
    ]
    out = build_research_sends({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [],
        "research_query": "main query",
    })
    assert isinstance(out, list)
    assert len(out) == 4  # sq1×2 + sq2×1 + sq3×1
    assert all(isinstance(s, Send) for s in out)

    nodes = [s.node for s in out]
    assert nodes == [
        "web_researcher", "academic_researcher",
        "code_researcher", "kb_researcher",
    ]

    # payload 完整
    for s in out:
        assert s.arg["research_query"] == "main query"
        assert s.arg["sub_question"] is not None


def test_need_more_research_re_fanout():
    plan = [SubQuestion(id="sq1", question="q1", recommended_sources=["web"])]
    out = supervisor_route({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [object()],  # 已有 evidence，但 reflector 要求补查
        "next_action": "need_more_research",
        "research_query": "x",
    })
    assert out == "research_subgraph"

    sends = build_research_sends({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [object()],
        "next_action": "need_more_research",
        "research_query": "x",
    })
    assert len(sends) == 1
    assert sends[0].node == "web_researcher"


def test_done_subquestion_skipped():
    plan = [
        SubQuestion(id="sq1", question="q1", recommended_sources=["web"], status="done"),
        SubQuestion(id="sq2", question="q2", recommended_sources=["web"]),
    ]
    out = build_research_sends({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [],
        "research_query": "x",
    })
    assert isinstance(out, list) and len(out) == 1
    assert out[0].arg["sub_question"].id == "sq2"
