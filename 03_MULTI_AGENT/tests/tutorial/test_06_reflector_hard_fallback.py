"""配套 TUTORIAL.md 第 6 章 —— Reflector 第 3 轮硬兜底

学什么：reflector_node 在 rc >= MAX_REVISION(=3) 时直接返回 force_complete，
        **不调用 LLM**。这是显式成本控制 + 防无限循环的关键防线。

断言：
1. revision_count=2 时（即将进入第 3 轮），reflector 不调用 get_llm
2. 返回 next_action=force_complete，revision_count=3
3. revision_count=0 时（首轮），LLM 必须被调用以做覆盖度评分
"""
from __future__ import annotations

import pytest

from agents.reflector import reflector_node
from agents.schemas import ReflectionResult, SubQuestion


class _SpyLLM:
    def __init__(self, reflection_payload=None):
        self.calls = 0
        self._payload = reflection_payload or ReflectionResult(
            coverage_by_subq={"sq1": 90}, missing_aspects=[], next_action="sufficient"
        )

    def with_structured_output(self, schema, **_kw):
        outer = self

        class _S:
            async def ainvoke(self, *_a, **_kw):
                outer.calls += 1
                return outer._payload

        return _S()


@pytest.fixture
def spy(monkeypatch):
    s = _SpyLLM()
    monkeypatch.setattr("agents.reflector.get_llm", lambda *a, **k: s)
    return s


@pytest.mark.asyncio
async def test_third_round_force_complete_no_llm_call(spy):
    state = {
        "plan": [SubQuestion(id="sq1", question="q", recommended_sources=["web"])],
        "evidence": [],
        "revision_count": 2,
    }
    out = await reflector_node(state)
    assert out["next_action"] == "force_complete"
    assert out["revision_count"] == 3
    assert spy.calls == 0, "硬兜底分支不应触发 LLM 调用"


@pytest.mark.asyncio
async def test_first_round_invokes_llm_for_scoring(spy):
    state = {
        "plan": [SubQuestion(id="sq1", question="q", recommended_sources=["web"])],
        "evidence": [],
        "revision_count": 0,
    }
    out = await reflector_node(state)
    assert spy.calls == 1
    assert out["next_action"] == "sufficient"
    assert out["coverage_by_subq"] == {"sq1": 90}
    assert out["revision_count"] == 1
