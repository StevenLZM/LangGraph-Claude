"""Production-grade eval runner for InsightLoop.

This command intentionally calls the real graph, real LLMs, configured search
tools, writer, judge, and report pipeline by delegating to evals.run. It adds
preflight checks and release-style gates around that live execution.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.settings import settings
from evals import run as eval_run

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductionGateConfig:
    min_success_rate: float = 1.0
    min_avg_overall: int = 80
    min_case_overall: int = 70
    min_avg_subquestion_recall: float = 0.7
    max_avg_elapsed_sec: float = 600.0
    max_high_risk_cases: int = 0
    max_tool_error_rate: float = 0.0
    max_llm_errors: int = 0


def _setting_value(source: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key)


def check_required_environment(source: Mapping[str, Any] | Any = settings) -> list[str]:
    """Return missing production prerequisites without exposing secret values."""
    missing: list[str] = []
    if not _setting_value(source, "deepseek_api_key"):
        missing.append("DEEPSEEK_API_KEY")

    search_keys = [
        _setting_value(source, "tavily_api_key"),
        _setting_value(source, "dashscope_api_key"),
        _setting_value(source, "brave_api_key"),
    ]
    if not any(search_keys):
        missing.append("one of TAVILY_API_KEY/DASHSCOPE_API_KEY/BRAVE_API_KEY")
    return missing


def build_run_args(*, dataset: str, limit: int) -> argparse.Namespace:
    """Build the namespace consumed by evals.run.main_async."""
    return argparse.Namespace(dataset=dataset, limit=limit)


def load_results(results_path: Path) -> list[dict[str, Any]]:
    records = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize production readiness signals from live eval records."""
    case_count = len(records)
    success = [r for r in records if not r.get("error")]
    scores = [
        int((r.get("score") or {}).get("overall"))
        for r in success
        if (r.get("score") or {}).get("overall") is not None
    ]
    subq_recalls = [
        float((r.get("retrieval_metrics") or {}).get("subquestion_recall"))
        for r in records
        if (r.get("retrieval_metrics") or {}).get("subquestion_recall") is not None
    ]
    elapsed = [
        float((r.get("runtime_health") or {}).get("elapsed_sec", r.get("elapsed_sec")))
        for r in records
        if (r.get("runtime_health") or {}).get("elapsed_sec", r.get("elapsed_sec")) is not None
    ]
    high_risk_count = sum(1 for r in records if (r.get("sampling") or {}).get("risk_level") == "high")
    manual_review_count = sum(1 for r in records if (r.get("sampling") or {}).get("review_action") == "manual_review")
    llm_calls = sum(int((r.get("node_metrics") or {}).get("llm_calls") or 0) for r in records)
    llm_errors = sum(int((r.get("node_metrics") or {}).get("llm_errors") or 0) for r in records)
    tool_calls = sum(int((r.get("node_metrics") or {}).get("tool_calls") or 0) for r in records)
    tool_errors = sum(int((r.get("node_metrics") or {}).get("tool_errors") or 0) for r in records)
    total_tokens = sum(int((r.get("node_metrics") or {}).get("total_tokens") or 0) for r in records)

    return {
        "case_count": case_count,
        "success_count": len(success),
        "failure_count": case_count - len(success),
        "success_rate": round(len(success) / case_count, 3) if case_count else 0.0,
        "avg_overall": _avg([float(v) for v in scores]),
        "min_overall": min(scores) if scores else None,
        "avg_subquestion_recall": _avg(subq_recalls),
        "avg_elapsed_sec": _avg(elapsed),
        "high_risk_count": high_risk_count,
        "manual_review_count": manual_review_count,
        "llm_calls": llm_calls,
        "llm_errors": llm_errors,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "tool_error_rate": round(tool_errors / tool_calls, 3) if tool_calls else 0.0,
        "total_tokens": total_tokens,
    }


def evaluate_production_gates(
    summary: dict[str, Any],
    config: ProductionGateConfig,
) -> list[str]:
    """Return gate failures; empty means production eval passed."""
    failures: list[str] = []

    if summary["success_rate"] < config.min_success_rate:
        failures.append(f"success_rate {summary['success_rate']} < {config.min_success_rate}")
    if summary["avg_overall"] is None or summary["avg_overall"] < config.min_avg_overall:
        failures.append(f"avg_overall {summary['avg_overall']} < {config.min_avg_overall}")
    if summary["min_overall"] is None or summary["min_overall"] < config.min_case_overall:
        failures.append(f"min_overall {summary['min_overall']} < {config.min_case_overall}")
    if (
        summary["avg_subquestion_recall"] is None
        or summary["avg_subquestion_recall"] < config.min_avg_subquestion_recall
    ):
        failures.append(
            f"avg_subquestion_recall {summary['avg_subquestion_recall']} "
            f"< {config.min_avg_subquestion_recall}"
        )
    if summary["avg_elapsed_sec"] is not None and summary["avg_elapsed_sec"] > config.max_avg_elapsed_sec:
        failures.append(f"avg_elapsed_sec {summary['avg_elapsed_sec']} > {config.max_avg_elapsed_sec:g}")
    if summary["high_risk_count"] > config.max_high_risk_cases:
        failures.append(f"high_risk_count {summary['high_risk_count']} > {config.max_high_risk_cases}")
    if summary.get("tool_error_rate", 0.0) > config.max_tool_error_rate:
        failures.append(f"tool_error_rate {summary.get('tool_error_rate')} > {config.max_tool_error_rate}")
    if summary.get("llm_errors", 0) > config.max_llm_errors:
        failures.append(f"llm_errors {summary.get('llm_errors')} > {config.max_llm_errors}")

    return failures


def write_production_report(
    *,
    out_dir: Path,
    summary: dict[str, Any],
    failures: list[str],
    config: ProductionGateConfig,
) -> Path:
    report = out_dir / "PRODUCTION_REPORT.md"
    status = "PASS" if not failures else "FAIL"
    gate_lines = "\n".join(f"- {failure}" for failure in failures) or "- none"
    body = (
        f"# InsightLoop Production Eval Gate — {status}\n\n"
        "## Summary\n"
        f"- case_count: **{summary['case_count']}**\n"
        f"- success_rate: **{summary['success_rate']}**\n"
        f"- avg_overall: **{summary['avg_overall']}**\n"
        f"- min_overall: **{summary['min_overall']}**\n"
        f"- avg_subquestion_recall: **{summary['avg_subquestion_recall']}**\n"
        f"- avg_elapsed_sec: **{summary['avg_elapsed_sec']}**\n"
        f"- high_risk_count: **{summary['high_risk_count']}**\n"
        f"- manual_review_count: **{summary['manual_review_count']}**\n\n"
        "## Trace Metrics\n"
        f"- llm_calls: **{summary['llm_calls']}**\n"
        f"- llm_errors: **{summary['llm_errors']}**\n"
        f"- tool_calls: **{summary['tool_calls']}**\n"
        f"- tool_errors: **{summary['tool_errors']}**\n"
        f"- tool_error_rate: **{summary['tool_error_rate']}**\n"
        f"- total_tokens: **{summary['total_tokens']}**\n\n"
        "## Gates\n"
        f"- min_success_rate: **{config.min_success_rate}**\n"
        f"- min_avg_overall: **{config.min_avg_overall}**\n"
        f"- min_case_overall: **{config.min_case_overall}**\n"
        f"- min_avg_subquestion_recall: **{config.min_avg_subquestion_recall}**\n"
        f"- max_avg_elapsed_sec: **{config.max_avg_elapsed_sec}**\n"
        f"- max_high_risk_cases: **{config.max_high_risk_cases}**\n"
        f"- max_tool_error_rate: **{config.max_tool_error_rate}**\n"
        f"- max_llm_errors: **{config.max_llm_errors}**\n\n"
        "## Gate Failures\n"
        f"{gate_lines}\n"
    )
    report.write_text(body, encoding="utf-8")
    return report


async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.skip_env_check:
        missing = check_required_environment()
        if missing:
            logger.error("[prod-eval] 缺少生产评测环境变量: %s", ", ".join(missing))
            return 2

    run_args = build_run_args(dataset=args.dataset, limit=args.limit)
    out_dir = await eval_run.main_async(run_args)
    records = load_results(out_dir / "results.jsonl")
    config = ProductionGateConfig(
        min_success_rate=args.min_success_rate,
        min_avg_overall=args.min_avg_overall,
        min_case_overall=args.min_case_overall,
        min_avg_subquestion_recall=args.min_avg_subquestion_recall,
        max_avg_elapsed_sec=args.max_avg_elapsed_sec,
        max_high_risk_cases=args.max_high_risk_cases,
        max_tool_error_rate=args.max_tool_error_rate,
        max_llm_errors=args.max_llm_errors,
    )
    summary = summarize_results(records)
    failures = evaluate_production_gates(summary, config)
    prod_report = write_production_report(
        out_dir=out_dir,
        summary=summary,
        failures=failures,
        config=config,
    )
    logger.info("[prod-eval] production report → %s", prod_report)
    if failures:
        logger.error("[prod-eval] gate failed: %s", "; ".join(failures))
        return 1
    logger.info("[prod-eval] gate passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InsightLoop production eval gate")
    parser.add_argument("--dataset", default=str(eval_run.DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=1, help="0=全部；生产烟测默认 1")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--min-avg-overall", type=int, default=80)
    parser.add_argument("--min-case-overall", type=int, default=70)
    parser.add_argument("--min-avg-subquestion-recall", type=float, default=0.7)
    parser.add_argument("--max-avg-elapsed-sec", type=float, default=600.0)
    parser.add_argument("--max-high-risk-cases", type=int, default=0)
    parser.add_argument("--max-tool-error-rate", type=float, default=0.0)
    parser.add_argument("--max-llm-errors", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
