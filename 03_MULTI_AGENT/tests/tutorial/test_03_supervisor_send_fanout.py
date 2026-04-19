"""配套 TUTORIAL.md 第 5 章 —— Send + Supervisor 路由

学什么：supervisor_route 依据 state 返回 3 种出口之一：
  · "planner" —— plan 未确认
  · "writer"  —— revision_count 超限 / 路径无 evidence 待补
  · list[Send] —— 真正的并行扇出，每个 Send 一个 researcher 任务

断言：
1. plan_confirmed=False → 回到 planner
2. revision_count >= 3 → 直接 writer
3. plan 已确认 + 无 evidence → 按 sub_question × recommended_sources 笛卡尔生成 Send
4. Send 节点名映射正确（web→web_researcher 等）
5. payload 包含 sub_question 与 research_query
"""
from __future__ import annotations

from langgraph.types import Send

from agents.schemas import SubQuestion
from graph.router import supervisor_route


def test_plan_not_confirmed_returns_planner():
    out = supervisor_route({"plan_confirmed": False})
    assert out == "planner"


def test_revision_overflow_goes_to_writer():
    out = supervisor_route({"plan_confirmed": True, "revision_count": 3})
    assert out == "writer"


def test_first_round_fans_out_send_list():
    plan = [
        SubQuestion(id="sq1", question="q1", recommended_sources=["web", "academic"]),
        SubQuestion(id="sq2", question="q2", recommended_sources=["code"]),
        SubQuestion(id="sq3", question="q3", recommended_sources=["kb"]),
    ]
    out = supervisor_route({
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
    assert isinstance(out, list) and len(out) == 1
    assert out[0].node == "web_researcher"


def test_done_subquestion_skipped():
    plan = [
        SubQuestion(id="sq1", question="q1", recommended_sources=["web"], status="done"),
        SubQuestion(id="sq2", question="q2", recommended_sources=["web"]),
    ]
    out = supervisor_route({
        "plan_confirmed": True,
        "plan": plan,
        "evidence": [],
        "research_query": "x",
    })
    assert isinstance(out, list) and len(out) == 1
    assert out[0].arg["sub_question"].id == "sq2"
