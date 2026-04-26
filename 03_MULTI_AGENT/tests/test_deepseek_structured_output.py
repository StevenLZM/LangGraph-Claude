from __future__ import annotations

import pytest

from agents.planner import planner_node
from agents.reflector import reflector_node
from agents.schemas import ReflectionResult, ResearchPlan, SubQuestion


class _RecordingStructured:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    async def ainvoke(self, messages, **_kwargs):
        self.messages = messages
        return self.payload


class _RecordingLLM:
    def __init__(self, payload):
        self.structured = _RecordingStructured(payload)
        self.method = None

    def with_structured_output(self, _schema, **kwargs):
        self.method = kwargs.get("method")
        return self.structured


def _message_text(messages) -> str:
    return "\n".join(getattr(m, "content", "") for m in messages)


@pytest.mark.asyncio
async def test_planner_uses_deepseek_json_mode(monkeypatch):
    plan = ResearchPlan(
        sub_questions=[
            SubQuestion(id="sq1", question="框架格局", recommended_sources=["web"]),
        ],
        estimated_depth="quick",
    )
    llm = _RecordingLLM(plan)
    monkeypatch.setattr("agents.planner.get_llm", lambda *args, **kwargs: llm)
    monkeypatch.setattr("agents.planner.interrupt", lambda _payload: None)

    await planner_node(
        {"research_query": "分析 2025 年开源 Agent 框架格局", "audience": "intermediate"}
    )

    assert llm.method == "json_mode"
    assert "json" in _message_text(llm.structured.messages).lower()


@pytest.mark.asyncio
async def test_reflector_uses_deepseek_json_mode(monkeypatch):
    reflection = ReflectionResult(
        coverage_by_subq={"sq1": 90},
        missing_aspects=[],
        next_action="sufficient",
    )
    llm = _RecordingLLM(reflection)
    monkeypatch.setattr("agents.reflector.get_llm", lambda *args, **kwargs: llm)

    await reflector_node(
        {
            "plan": [SubQuestion(id="sq1", question="框架格局", recommended_sources=["web"])],
            "evidence": [],
            "revision_count": 0,
        }
    )

    assert llm.method == "json_mode"
    assert "json" in _message_text(llm.structured.messages).lower()
