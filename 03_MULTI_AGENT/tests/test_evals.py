"""evals/* 单测 —— mock judge LLM + 给定 results.jsonl 验证 report 渲染。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.diagnostics import (
    build_component_quality,
    build_process_quality,
    build_retrieval_metrics,
    build_runtime_health,
    build_sampling_decision,
)
from evals.frameworks import build_framework_coverage, framework_matrix_markdown
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


def test_process_quality_summarizes_plan_evidence_and_reflection_state():
    plan = [
        {"id": "sq1", "question": "A", "recommended_sources": ["web"]},
        {"id": "sq2", "question": "B", "recommended_sources": ["kb"]},
    ]
    evidence = [
        {"sub_question_id": "sq1", "source_type": "web", "source_url": "https://a", "snippet": "a"},
        {"sub_question_id": "sq1", "source_type": "kb", "source_url": "kb://b", "snippet": "b"},
    ]
    state = {
        "revision_count": 3,
        "next_action": "force_complete",
        "missing_aspects": ["B 缺少证据"],
        "citation_audit_issues": ["unknown citation [^9]; evidence_count=2"],
    }

    out = build_process_quality(plan=plan, evidence=evidence, state=state)

    assert out["plan_size"] == 2
    assert out["evidence_count"] == 2
    assert out["evidence_per_subq"] == {"sq1": 2, "sq2": 0}
    assert out["missing_subquestions"] == ["sq2"]
    assert out["source_counts"] == {"web": 1, "kb": 1}
    assert out["source_diversity"] == 2
    assert out["forced_completion"] is True
    assert out["citation_audit_issue_count"] == 1


def test_retrieval_metrics_cover_rag_style_proxy_metrics():
    plan = [
        {"id": "sq1", "question": "A", "recommended_sources": ["web", "kb"]},
        {"id": "sq2", "question": "B", "recommended_sources": ["academic"]},
        {"id": "sq3", "question": "C", "recommended_sources": ["code"]},
    ]
    evidence = [
        {"sub_question_id": "sq1", "source_type": "web", "source_url": "https://a", "snippet": "a"},
        {"sub_question_id": "sq2", "source_type": "academic", "source_url": "https://b", "snippet": "b"},
    ]

    out = build_retrieval_metrics(plan=plan, evidence=evidence)

    assert out["planned_subquestion_count"] == 3
    assert out["covered_subquestion_count"] == 2
    assert out["subquestion_recall"] == 0.667
    assert out["planned_source_types"] == ["academic", "code", "kb", "web"]
    assert out["covered_source_types"] == ["academic", "web"]
    assert out["source_type_recall"] == 0.5
    assert out["evidence_density"] == 0.67


def test_component_quality_scores_planner_research_reflector_writer_and_runtime():
    process_quality = {
        "plan_size": 3,
        "evidence_count": 2,
        "missing_subquestions": ["sq3"],
        "forced_completion": True,
        "citation_audit_issue_count": 1,
    }
    retrieval_metrics = {"subquestion_recall": 0.667}
    runtime_health = {"status": "ok", "slow_case": True}
    score = {"citation": 72}

    out = build_component_quality(
        process_quality=process_quality,
        retrieval_metrics=retrieval_metrics,
        runtime_health=runtime_health,
        score=score,
    )

    assert out["planner"]["score"] == 100
    assert out["research"]["score"] == 67
    assert out["reflector"]["status"] == "warn"
    assert out["writer"]["score"] == 47
    assert out["runtime"]["score"] == 80


def test_sampling_decision_flags_low_score_process_gaps_and_runtime_errors():
    process_quality = {
        "evidence_count": 2,
        "plan_size": 3,
        "missing_subquestions": ["sq2"],
        "forced_completion": True,
        "citation_audit_issue_count": 1,
    }
    runtime_health = build_runtime_health(error="RuntimeError: tool failed", elapsed_sec=301.2)
    score = {"overall": 68, "coverage": 70, "accuracy": 60, "citation": 65}

    out = build_sampling_decision(
        score=score,
        process_quality=process_quality,
        runtime_health=runtime_health,
    )

    assert out["risk_level"] == "high"
    assert out["review_action"] == "manual_review"
    assert out["reasons"] == [
        "execution_error",
        "low_overall_score",
        "citation_audit_issue",
        "missing_subquestion_evidence",
        "forced_completion",
        "slow_case",
    ]


def test_framework_coverage_explains_teaching_eval_frameworks():
    record = {
        "case": {"id": "tech_01"},
        "score": {"overall": 82},
        "process_quality": {"plan_size": 2},
        "retrieval_metrics": {"subquestion_recall": 1.0},
        "component_quality": {"planner": {"score": 100}},
        "runtime_health": {"status": "ok"},
        "sampling": {"risk_level": "low"},
    }

    out = build_framework_coverage(record)
    by_id = {item["id"]: item for item in out}

    assert by_id["llm_as_judge"]["status"] == "covered"
    assert by_id["retrieval_proxy"]["status"] == "covered"
    assert by_id["component_quality"]["status"] == "covered"
    assert by_id["human_review_rubric"]["status"] == "template"
    assert by_id["trace_observability"]["status"] == "partial"

    md = framework_matrix_markdown(out)
    assert "教学评估框架覆盖矩阵" in md
    assert "LLM-as-judge" in md
    assert "RAG/Retrieval proxy" in md
    assert "Human review rubric" in md
    assert "Trace observability" in md


def test_render_markdown_includes_three_layer_eval_sections(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    row = {
        "case": {"id": "tech_01", "category": "技术", "query": "..."},
        "score": {"coverage": 85, "accuracy": 80, "citation": 70, "overall": 79, "rationale": "ok"},
        "process_quality": {
            "plan_size": 2,
            "evidence_count": 2,
            "source_diversity": 1,
            "missing_subquestions": ["sq2"],
            "forced_completion": False,
            "citation_audit_issue_count": 0,
        },
        "runtime_health": {"status": "ok", "elapsed_sec": 88.0, "slow_case": False},
        "retrieval_metrics": {"subquestion_recall": 0.5, "source_type_recall": 1.0, "evidence_density": 1.0},
        "component_quality": {
            "planner": {"score": 100, "status": "pass"},
            "research": {"score": 50, "status": "warn"},
            "reflector": {"score": 100, "status": "pass"},
            "writer": {"score": 70, "status": "warn"},
            "runtime": {"score": 100, "status": "pass"},
        },
        "node_metrics": {
            "llm_calls": 4,
            "tool_calls": 6,
            "tool_errors": 1,
            "total_tokens": 1200,
            "total_elapsed_ms": 88000.0,
        },
        "sampling": {
            "risk_level": "medium",
            "review_action": "spot_check",
            "reasons": ["medium_overall_score", "missing_subquestion_evidence"],
        },
        "elapsed_sec": 88.0,
        "error": None,
    }
    results.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    out = render_markdown(results, tmp_path / "REPORT.md", run_id="run-x")
    text = out.read_text(encoding="utf-8")

    assert "三层评估覆盖" in text
    assert "过程链路质量" in text
    assert "运行稳定性" in text
    assert "组件质量评分" in text
    assert "RAG/Retrieval Proxy" in text
    assert "Trace Metrics" in text
    assert "人工复核队列" in text
    assert "教学评估框架覆盖矩阵" in text
    assert "抽检建议" in text
    assert "medium" in text
    assert "missing_subquestion_evidence" in text


def test_evals_ui_load_run_flattens_diagnostic_fields(tmp_path: Path, monkeypatch):
    from app import evals_ui

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    row = {
        "case": {"id": "tech_01", "category": "技术", "query": "q", "audience": "expert"},
        "score": {"coverage": 90, "accuracy": 80, "citation": 70, "overall": 81, "rationale": "ok"},
        "process_quality": {
            "plan_size": 2,
            "evidence_count": 5,
            "source_diversity": 3,
            "missing_subquestions": ["sq2"],
            "forced_completion": True,
            "citation_audit_issue_count": 1,
        },
        "runtime_health": {"status": "ok", "elapsed_sec": 99.0, "slow_case": False},
        "retrieval_metrics": {"subquestion_recall": 1.0, "source_type_recall": 0.75},
        "component_quality": {
            "planner": {"score": 100},
            "research": {"score": 100},
            "reflector": {"score": 100},
            "writer": {"score": 75},
            "runtime": {"score": 100},
        },
        "node_metrics": {
            "llm_calls": 4,
            "tool_calls": 6,
            "tool_errors": 1,
            "total_tokens": 1200,
        },
        "sampling": {
            "risk_level": "high",
            "review_action": "manual_review",
            "reasons": ["forced_completion"],
        },
        "elapsed_sec": 99.0,
        "error": None,
    }
    (run_dir / "results.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setattr(evals_ui, "RESULTS_ROOT", tmp_path)

    df = evals_ui._load_run("run-1")

    assert df.loc[0, "process_evidence_count"] == 5
    assert df.loc[0, "source_diversity"] == 3
    assert df.loc[0, "risk_level"] == "high"
    assert df.loc[0, "review_action"] == "manual_review"
    assert df.loc[0, "sampling_reasons"] == "forced_completion"
    assert df.loc[0, "subquestion_recall"] == 1.0
    assert df.loc[0, "writer_score"] == 75
    assert df.loc[0, "llm_calls"] == 4
    assert df.loc[0, "total_tokens"] == 1200


def test_eval_readme_documents_teaching_frameworks():
    text = (Path(__file__).resolve().parent.parent / "evals" / "README.md").read_text(encoding="utf-8")

    assert "三层评估" in text
    assert "PRODUCTION_EVALS_TEACHING.md" in text
    assert "LLM-as-judge" in text
    assert "RAG/Retrieval proxy" in text
    assert "Human review rubric" in text
    assert "Trace observability" in text
    assert "Run-to-run / A-B comparison" in text
    assert "node_metrics" in text
    assert "manual_review_queue.jsonl" in text
    assert "LANGSMITH_API_KEY" in text


def test_production_evals_teaching_doc_covers_full_flow_and_interview_questions():
    text = (
        Path(__file__).resolve().parent.parent
        / "evals"
        / "PRODUCTION_EVALS_TEACHING.md"
    ).read_text(encoding="utf-8")

    assert "Production-Grade Agent Evals 教学文档" in text
    assert "为什么 Agent Evals 要分三层" in text
    assert "Callback Trace 和 LangSmith" in text
    assert "无用户反馈下的抽检设计" in text
    assert "人工复核数据闭环" in text
    assert "生产门禁" in text
    assert "面试问题与期望回答" in text
    assert "Q1：为什么 Agent 系统不能只用最终答案分做评估？" in text
    assert "期望回答" in text
