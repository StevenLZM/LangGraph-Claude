"""端到端离线闭环测试 —— 打桩 LLM + registry，验证图从 planner 经 interrupt 恢复到 writer 全跑通。"""
from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents.schemas import ReflectionResult, ResearchPlan, SubQuestion
from graph.workflow import build_graph


class _FakeStructuredLLM:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, *args, **kwargs):
        return self._payload

    async def ainvoke(self, *args, **kwargs):
        return self._payload


class _FakeLLM:
    """模拟 LangChain ChatModel：with_structured_output + ainvoke（写报告）"""

    def __init__(self, *, plan=None, reflection=None, report_md=None):
        self._plan = plan
        self._reflection = reflection
        self._report_md = report_md or "# Stub Report\n\n正文。"

    def with_structured_output(self, schema, **kwargs):
        if schema is ResearchPlan:
            return _FakeStructuredLLM(self._plan)
        if schema is ReflectionResult:
            return _FakeStructuredLLM(self._reflection)
        raise ValueError(f"unexpected schema {schema}")

    async def ainvoke(self, *args, **kwargs):
        class _Msg:
            content = self._report_md

        return _Msg()


@pytest.fixture
def patched(monkeypatch):
    """打桩 get_llm + registry，屏蔽真实外部依赖。"""
    plan = ResearchPlan(
        sub_questions=[SubQuestion(id="sq1", question="test subq", recommended_sources=["web"])],
        estimated_depth="quick",
    )
    reflection = ReflectionResult(
        coverage_by_subq={"sq1": 85}, missing_aspects=[], next_action="sufficient"
    )
    fake = _FakeLLM(plan=plan, reflection=reflection, report_md="# 报告\n\n## 结论\n\n结论内容。")

    monkeypatch.setattr("agents.planner.get_llm", lambda *a, **k: fake)
    monkeypatch.setattr("agents.reflector.get_llm", lambda *a, **k: fake)
    monkeypatch.setattr("agents.writer.get_llm", lambda *a, **k: fake)

    # Registry 中塞一个 fake web 工具
    class _FakeWeb:
        name = "fake-web"
        source_type = "web"

        async def search(self, q, *, top_k=5):
            return [{"snippet": f"fact about {q}", "source_url": f"https://fake/{q}", "relevance_score": 0.8}]

        async def close(self):
            pass

    from tools.registry import ToolRegistry
    from app import bootstrap

    reg = ToolRegistry()
    reg.register(_FakeWeb())
    bootstrap.app_state.registry = reg

    yield


@pytest.mark.asyncio
async def test_interrupt_resume_full_flow(patched, tmp_path):
    from config.settings import settings as _s
    _s.reports_dir = str(tmp_path)

    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke(
        {"research_query": "AI 行业测试", "audience": "intermediate", "messages": [], "evidence": []},
        config=cfg,
    )
    intr = first.get("__interrupt__")
    assert intr, "Planner 应触发 interrupt"

    edited = {
        "sub_questions": [
            {"id": "sq1", "question": "修改后的子问题", "recommended_sources": ["web"], "status": "pending"}
        ],
        "estimated_depth": "quick",
    }
    result = await graph.ainvoke(Command(resume={"plan": edited}), config=cfg)
    assert result.get("final_report"), "应生成 final_report"
    assert "报告" in result["final_report"] or "结论" in result["final_report"]
    assert result.get("evidence"), "应有 evidence（来自 fake-web）"
    assert result["evidence"][0].source_url.startswith("https://fake/")
