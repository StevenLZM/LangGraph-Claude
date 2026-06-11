"""Production eval framework tests.

These tests validate the production gate logic without calling external LLMs or
tools. The live graph/LLM smoke test is opt-in via INSIGHTLOOP_RUN_LIVE_EVAL=1.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from evals.production import (
    ProductionGateConfig,
    build_run_args,
    check_required_environment,
    evaluate_production_gates,
    summarize_results,
    write_production_report,
)


def test_required_environment_requires_llm_and_one_live_search_provider():
    missing = check_required_environment(
        {
            "deepseek_api_key": "",
            "tavily_api_key": "",
            "dashscope_api_key": "",
            "brave_api_key": "",
        }
    )

    assert missing == ["DEEPSEEK_API_KEY", "one of TAVILY_API_KEY/DASHSCOPE_API_KEY/BRAVE_API_KEY"]

    assert check_required_environment(
        {
            "deepseek_api_key": "sk-live",
            "tavily_api_key": "",
            "dashscope_api_key": "sk-search",
            "brave_api_key": "",
        }
    ) == []


def test_summarize_results_includes_success_quality_retrieval_runtime_and_risk():
    records = [
        {
            "error": None,
            "score": {"overall": 85},
            "retrieval_metrics": {"subquestion_recall": 0.75},
            "runtime_health": {"elapsed_sec": 120.0},
            "node_metrics": {"llm_calls": 2, "llm_errors": 0, "tool_calls": 5, "tool_errors": 1, "total_tokens": 100},
            "sampling": {"risk_level": "low", "review_action": "random_sample"},
        },
        {
            "error": "RuntimeError: failed",
            "score": None,
            "retrieval_metrics": {"subquestion_recall": 0.0},
            "runtime_health": {"elapsed_sec": 30.0},
            "node_metrics": {"llm_calls": 1, "llm_errors": 1, "tool_calls": 3, "tool_errors": 0, "total_tokens": 20},
            "sampling": {"risk_level": "high", "review_action": "manual_review"},
        },
    ]

    out = summarize_results(records)

    assert out["case_count"] == 2
    assert out["success_count"] == 1
    assert out["failure_count"] == 1
    assert out["success_rate"] == 0.5
    assert out["avg_overall"] == 85.0
    assert out["min_overall"] == 85
    assert out["avg_subquestion_recall"] == 0.375
    assert out["avg_elapsed_sec"] == 75.0
    assert out["llm_calls"] == 3
    assert out["llm_errors"] == 1
    assert out["tool_calls"] == 8
    assert out["tool_errors"] == 1
    assert out["tool_error_rate"] == 0.125
    assert out["total_tokens"] == 120
    assert out["high_risk_count"] == 1
    assert out["manual_review_count"] == 1


def test_evaluate_production_gates_reports_actionable_failures():
    summary = {
        "case_count": 2,
        "success_rate": 0.5,
        "avg_overall": 78.0,
        "min_overall": 62,
        "avg_subquestion_recall": 0.55,
        "avg_elapsed_sec": 700.0,
        "high_risk_count": 1,
        "tool_error_rate": 0.2,
        "llm_errors": 1,
    }
    config = ProductionGateConfig(
        min_success_rate=1.0,
        min_avg_overall=80,
        min_case_overall=70,
        min_avg_subquestion_recall=0.7,
        max_avg_elapsed_sec=600,
        max_high_risk_cases=0,
        max_tool_error_rate=0.0,
        max_llm_errors=0,
    )

    failures = evaluate_production_gates(summary, config)

    assert failures == [
        "success_rate 0.5 < 1.0",
        "avg_overall 78.0 < 80",
        "min_overall 62 < 70",
        "avg_subquestion_recall 0.55 < 0.7",
        "avg_elapsed_sec 700.0 > 600",
        "high_risk_count 1 > 0",
        "tool_error_rate 0.2 > 0.0",
        "llm_errors 1 > 0",
    ]


def test_build_run_args_preserves_dataset_and_limit():
    args = build_run_args(dataset="evals/dataset.jsonl", limit=1)

    assert args.dataset == "evals/dataset.jsonl"
    assert args.limit == 1


def test_write_production_report_includes_trace_gates(tmp_path: Path):
    summary = {
        "case_count": 1,
        "success_rate": 1.0,
        "avg_overall": 88.0,
        "min_overall": 88,
        "avg_subquestion_recall": 1.0,
        "avg_elapsed_sec": 45.0,
        "high_risk_count": 0,
        "manual_review_count": 0,
        "llm_calls": 3,
        "llm_errors": 0,
        "tool_calls": 6,
        "tool_errors": 0,
        "tool_error_rate": 0.0,
        "total_tokens": 2048,
    }
    config = ProductionGateConfig(max_tool_error_rate=0.0, max_llm_errors=0)

    report = write_production_report(
        out_dir=tmp_path,
        summary=summary,
        failures=[],
        config=config,
    )

    text = report.read_text(encoding="utf-8")
    assert "Trace Metrics" in text
    assert "llm_calls: **3**" in text
    assert "tool_error_rate: **0.0**" in text
    assert "max_tool_error_rate: **0.0**" in text
    assert "max_llm_errors: **0**" in text


def test_makefile_and_eval_readme_document_production_runner():
    root = Path(__file__).resolve().parent.parent
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    readme = (root / "evals" / "README.md").read_text(encoding="utf-8")

    assert "eval-prod-smoke:" in makefile
    assert "eval-prod:" in makefile
    assert "python -m evals.production" in readme
    assert "PRODUCTION_REPORT.md" in readme
    assert "manual_review_queue.jsonl" in readme
    assert "tool error rate" in readme
    assert "INSIGHTLOOP_RUN_LIVE_EVAL=1" in readme


@pytest.mark.skipif(
    os.getenv("INSIGHTLOOP_RUN_LIVE_EVAL") != "1",
    reason="live production eval is opt-in; set INSIGHTLOOP_RUN_LIVE_EVAL=1",
)
def test_live_production_eval_smoke():
    """Opt-in live smoke test: calls real graph, real LLM, and real tools."""
    from evals.production import main

    rc = main(["--limit", "1"])

    assert rc == 0
