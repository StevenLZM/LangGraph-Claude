"""evals/* 单测 —— mock judge LLM + 给定 results.jsonl 验证 report 渲染。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.judge import JudgeInput, JudgeScore, build_judge_prompt, judge_one
from evals.report import render_markdown


def test_build_judge_prompt_contains_all_blocks():
    inp = JudgeInput(
        query="对比 A 和 B",
        plan=[{"id": "sq1", "question": "A 是什么？", "recommended_sources": ["web"]}],
        evidence_brief=[{"source_type": "web", "source_url": "https://x", "snippet": "A 是 ..."}],
        report_md="# 报告\n\nA 比 B 更好[^1]\n",
    )
    text = build_judge_prompt(inp)
    assert "原始研究问题" in text and "对比 A 和 B" in text
    assert "Planner 子问题" in text and "sq1" in text
    assert "Evidence 摘要" in text and "https://x" in text
    assert "报告全文" in text


def test_build_judge_prompt_truncates_long_report():
    long_md = "x" * 10000
    inp = JudgeInput(query="q", plan=[], evidence_brief=[], report_md=long_md)
    text = build_judge_prompt(inp, report_max_chars=100)
    assert "报告已截断" in text
    assert "10000 字" in text


class _FakeStructured:
    def __init__(self, payload):
        self._p = payload

    async def ainvoke(self, *args, **kwargs):
        return self._p


class _FakeJudgeLLM:
    def __init__(self, payload):
        self._p = payload

    def with_structured_output(self, schema, **kwargs):
        assert schema is JudgeScore
        assert kwargs.get("method") == "json_mode"
        return _FakeStructured(self._p)


@pytest.mark.asyncio
async def test_judge_one_returns_score():
    fake = _FakeJudgeLLM(JudgeScore(coverage=80, accuracy=70, citation=60, overall=72, rationale="ok"))
    out = await judge_one(
        JudgeInput(query="q", plan=[], evidence_brief=[], report_md="r"),
        llm=fake,
    )
    assert out.overall == 72 and out.rationale == "ok"


def test_render_markdown(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    rows = [
        {
            "case": {"id": "tech_01", "category": "技术", "query": "..."},
            "score": {"coverage": 85, "accuracy": 80, "citation": 70, "overall": 79, "rationale": "ok"},
            "elapsed_sec": 142.0,
            "report_path": "/tmp/x.md",
            "evidence_count": 12,
            "error": None,
        },
        {
            "case": {"id": "industry_01", "category": "产业", "query": "..."},
            "score": {"coverage": 60, "accuracy": 55, "citation": 50, "overall": 55, "rationale": "缺失关键数据"},
            "elapsed_sec": 89.0,
            "report_path": "/tmp/y.md",
            "evidence_count": 4,
            "error": None,
        },
        {
            "case": {"id": "broken_01", "category": "对比", "query": "..."},
            "score": None,
            "elapsed_sec": 5.0,
            "error": "RuntimeError: planner 未触发 interrupt",
        },
    ]
    with results.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = render_markdown(results, tmp_path / "REPORT.md", run_id="20260426-153000")
    text = out.read_text(encoding="utf-8")
    assert "20260426-153000" in text
    assert "用例总数: **3**" in text and "成功: **2**" in text
    assert "tech_01" in text and "industry_01" in text and "broken_01" in text
    assert "维度均值" in text
    assert "**72.5**" in text or "72.5" in text  # (85+60)/2 = 72.5 coverage avg
    assert "失分案例" in text
    assert "industry_01（综合 55）" in text
    assert "❌ 执行失败" in text
