"""配套 TUTORIAL.md 第 4 章 —— interrupt() + Command(resume=...) HITL

学什么：planner_node 调 interrupt({...}) 后图暂停，ainvoke 返回值的 __interrupt__
        字段携带 payload 给应用层；用户编辑后通过 Command(resume={...}) 第二次 ainvoke，
        decision 会作为 interrupt 的返回值进入节点继续执行。

断言：
1. 第一次 ainvoke：返回值含 __interrupt__，且 payload 是 planner 生成的 plan
2. 用户用 Command(resume={"plan": <edited>}) 恢复后，state.plan = 用户编辑版本
3. plan_confirmed 被置 True，后续路由可正常前进
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from agents.planner import planner_node
from agents.schemas import ResearchPlan, SubQuestion
from graph.state import ResearchState


class _FakeStructured:
    def __init__(self, payload):
        self._payload = payload

    async def ainvoke(self, *_a, **_kw):
        return self._payload


class _FakeLLM:
    def __init__(self, plan):
        self._plan = plan

    def with_structured_output(self, schema, **_kw):
        return _FakeStructured(self._plan)


@pytest.fixture
def patched_planner(monkeypatch):
    plan = ResearchPlan(
        sub_questions=[
            SubQuestion(id="sq1", question="原始问题 A", recommended_sources=["web"]),
            SubQuestion(id="sq2", question="原始问题 B", recommended_sources=["academic"]),
        ],
        estimated_depth="standard",
    )
    fake = _FakeLLM(plan)
    monkeypatch.setattr("agents.planner.get_llm", lambda *a, **k: fake)
    return plan


@pytest.fixture
def mini_graph():
    """只含 planner 节点的最小图，便于聚焦 HITL 行为。"""
    wf = StateGraph(ResearchState)
    wf.add_node("planner", planner_node)
    wf.set_entry_point("planner")
    wf.add_edge("planner", END)
    return wf.compile(checkpointer=MemorySaver())


@pytest.mark.asyncio
async def test_interrupt_payload_carries_plan(patched_planner, mini_graph):
    cfg = {"configurable": {"thread_id": "tutorial-05-a"}}
    first = await mini_graph.ainvoke(
        {"research_query": "X", "audience": "intermediate", "messages": [], "evidence": []},
        config=cfg,
    )
    intr = first.get("__interrupt__")
    assert intr, "planner 必须触发 interrupt"

    item = intr[0] if isinstance(intr, list) else intr
    payload = getattr(item, "value", None) or item.get("value")
    assert payload["phase"] == "plan_review"
    assert len(payload["plan"]["sub_questions"]) == 2


@pytest.mark.asyncio
async def test_resume_with_user_edit_overrides_plan(patched_planner, mini_graph):
    cfg = {"configurable": {"thread_id": "tutorial-05-b"}}
    await mini_graph.ainvoke(
        {"research_query": "X", "audience": "intermediate", "messages": [], "evidence": []},
        config=cfg,
    )

    edited = {
        "sub_questions": [
            {"id": "sq1", "question": "用户改写后的问题",
             "recommended_sources": ["kb"], "status": "pending"}
        ],
        "estimated_depth": "quick",
    }
    resumed = await mini_graph.ainvoke(Command(resume={"plan": edited}), config=cfg)

    assert resumed["plan_confirmed"] is True
    assert len(resumed["plan"]) == 1
    assert resumed["plan"][0].question == "用户改写后的问题"
    assert resumed["plan"][0].recommended_sources == ["kb"]
