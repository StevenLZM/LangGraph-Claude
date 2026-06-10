"""Writer citation audit behavior without real LLM calls."""
from __future__ import annotations

import pytest

from agents.schemas import Evidence, SubQuestion
from agents import writer


class _FakeLLM:
    async def ainvoke(self, _messages):
        class _Resp:
            content = "# 报告\n\n结论一[^1]，结论二[^2]。\n\n## 引用\n[^1]: https://a.example"

        return _Resp()


@pytest.mark.asyncio
async def test_writer_appends_missing_reference_entries(monkeypatch):
    monkeypatch.setattr(writer, "get_llm", lambda *_args, **_kwargs: _FakeLLM())
    monkeypatch.setattr(writer.report_store, "save", lambda *_args, **_kwargs: "/tmp/report.md")

    out = await writer.writer_node(
        {
            "research_query": "q",
            "audience": "intermediate",
            "plan": [
                SubQuestion(id="sq1", question="q1", recommended_sources=["web"]),
            ],
            "evidence": [
                Evidence(
                    sub_question_id="sq1",
                    source_type="web",
                    source_url="https://a.example",
                    snippet="a",
                ),
                Evidence(
                    sub_question_id="sq1",
                    source_type="web",
                    source_url="https://b.example",
                    snippet="b",
                ),
            ],
        },
        {"configurable": {"thread_id": "tid"}},
    )

    assert "[^2]: https://b.example" in out["final_report"]
    assert out["citation_audit_issues"] == []


class _FakeBadCitationLLM:
    async def ainvoke(self, _messages):
        class _Resp:
            content = "# 报告\n\n错误引用[^99]。"

        return _Resp()


@pytest.mark.asyncio
async def test_writer_returns_structured_citation_audit_issues(monkeypatch):
    monkeypatch.setattr(writer, "get_llm", lambda *_args, **_kwargs: _FakeBadCitationLLM())
    monkeypatch.setattr(writer.report_store, "save", lambda *_args, **_kwargs: "/tmp/report.md")

    out = await writer.writer_node(
        {
            "research_query": "q",
            "audience": "intermediate",
            "plan": [SubQuestion(id="sq1", question="q1", recommended_sources=["web"])],
            "evidence": [
                Evidence(
                    sub_question_id="sq1",
                    source_type="web",
                    source_url="https://a.example",
                    snippet="a",
                )
            ],
        },
        {"configurable": {"thread_id": "tid"}},
    )

    assert out["citation_audit_issues"] == ["unknown citation [^99]; evidence_count=1"]
