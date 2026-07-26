from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

from evals.ragas_adapter import RAGAS_METRIC_NAMES
from evals.statistics import bootstrap_mean_ci, paired_bootstrap_delta
from rag.retrieval_trace import REQUIRED_EVAL_STAGES


STAGE_HIGHLIGHTS: tuple[tuple[str, str, str], ...] = (
    ("bm25_child", "recall_at_k", "BM25 Recall@50"),
    ("dense_child", "recall_at_k", "Dense Recall@50"),
    ("rrf_child", "recall_at_k", "RRF Recall@80"),
    ("rrf_child", "ndcg_at_k", "RRF NDCG@80"),
    ("cross_encoder_child", "ndcg_at_k", "Cross-Encoder NDCG@15"),
    ("cross_encoder_child", "mrr_at_k", "Cross-Encoder MRR@15"),
    ("business_fused_child", "ndcg_at_k", "Business Fusion NDCG@15"),
    ("diversified_parent", "recall_at_k", "Diversified Parent Recall@8"),
    ("diversified_parent", "ndcg_at_k", "Diversified Parent NDCG@8"),
    ("final_context_parent", "recall_at_k", "Final Context Recall@6"),
    ("final_context_parent", "ndcg_at_k", "Final Context NDCG@6"),
    ("final_context_parent", "hit_at_k", "Final Context Hit@6"),
)

RAGAS_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Answer Correctness",
    "answer_relevancy": "Answer Relevancy",
    "semantic_similarity": "Semantic Similarity",
}


def publish_completed_run(
    run_dir: Path,
    *,
    baseline_run: Path | None = None,
) -> None:
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    results = _read_jsonl(run_dir / "case_results.partial.jsonl")
    _validate_publishable(manifest, results)

    case_results_path = run_dir / "case_results.jsonl"
    _write_jsonl(case_results_path, results)
    _write_stage_metrics(run_dir / "stage_metrics.jsonl", results)
    _write_ragas_results(run_dir / "ragas_results.jsonl", results)

    summary = summarize_results(results)
    summary["release_gates"] = _release_gates(
        manifest,
        results,
        baseline_run=baseline_run,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "REPORT.md").write_text(
        _build_report(manifest, summary, baseline_run=baseline_run),
        encoding="utf-8",
    )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions = {
        "global": {"all": _summarize_group(results)},
        "split": _group_summaries(results, lambda row: row.get("split", "unknown")),
        "category": _group_summaries(
            results,
            lambda row: row.get("category", "unknown"),
        ),
        "query_type": _group_summaries(
            results,
            lambda row: row.get("query_type", "unknown"),
        ),
        "expected_behavior": _group_summaries(
            results,
            lambda row: row.get("expected_behavior", "unknown"),
        ),
    }
    return {
        "total": len(results),
        "degraded_case_count": sum(
            bool(row.get("degraded_stages"))
            for row in results
        ),
        "permission_leakage_total": sum(
            int(row.get("behavior_metrics", {}).get("permission_leakage") or 0)
            for row in results
        ),
        "dimensions": dimensions,
    }


def _summarize_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    for stage_name in REQUIRED_EVAL_STAGES:
        for metric_name in ("recall_at_k", "ndcg_at_k", "mrr_at_k", "hit_at_k"):
            key = f"{stage_name}.{metric_name}"
            metrics[key] = _metric_summary(
                _finite_values(
                    row.get("stage_metrics", {})
                    .get(stage_name, {})
                    .get(metric_name)
                    for row in results
                )
            )
    for metric_name in RAGAS_METRIC_NAMES:
        metrics[f"ragas.{metric_name}"] = _metric_summary(
            _finite_values(
                row.get("ragas_metrics", {}).get(metric_name)
                for row in results
            )
        )
    for metric_name in ("abstain_accuracy", "deny_accuracy"):
        metrics[f"behavior.{metric_name}"] = _metric_summary(
            _finite_values(
                row.get("behavior_metrics", {}).get(metric_name)
                for row in results
            )
        )
    metrics["latency_ms"] = _metric_summary(
        _finite_values(row.get("latency_ms") for row in results)
    )
    return {"total": len(results), "metrics": metrics}


def _group_summaries(
    results: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(key(result)), []).append(result)
    return {
        name: _summarize_group(items)
        for name, items in sorted(grouped.items())
    }


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"average": None, "count": 0, "ci95": None}
    interval = bootstrap_mean_ci(values)
    return {
        "average": interval.estimate,
        "count": len(values),
        "ci95": {
            "lower": interval.lower,
            "upper": interval.upper,
        },
    }


def _validate_publishable(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    if manifest.get("status") != "completed":
        raise ValueError("只有 status=completed 的真实评测运行可以发布")
    if int(manifest.get("failed_case_count", 0)) != 0:
        raise ValueError("存在失败 Case，不能发布")
    expected = int(manifest.get("expected_case_count", -1))
    completed = int(manifest.get("completed_case_count", -1))
    if expected != completed or completed != len(results):
        raise ValueError("预期 Case、完成 Case 与结果文件数量不一致")
    if manifest.get("release_mode") and len(results) < 200:
        raise ValueError("正式 Test 发布至少需要 200 个已完成 Case")
    for result in results:
        if result.get("expected_behavior") != "answer":
            continue
        stages = result.get("stage_metrics") or {}
        missing = [name for name in REQUIRED_EVAL_STAGES if name not in stages]
        if missing:
            raise ValueError(
                f"Case {result.get('id')} 缺少阶段指标: {', '.join(missing)}"
            )


def _release_gates(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    baseline_run: Path | None,
) -> dict[str, Any]:
    safety_checks = {
        "all_cases_completed": (
            int(manifest.get("expected_case_count", -1))
            == int(manifest.get("completed_case_count", -2))
            == len(results)
        ),
        "permission_leakage_zero": all(
            int(row.get("behavior_metrics", {}).get("permission_leakage") or 0) == 0
            for row in results
        ),
        "no_degraded_cases": all(
            not row.get("degraded_stages")
            for row in results
        ),
    }
    if baseline_run is None:
        return {
            "status": "baseline_not_provided",
            "passed": None,
            "checks": safety_checks,
            "comparisons": {},
        }

    baseline_manifest = _read_json(Path(baseline_run) / "run_manifest.json")
    baseline_results = _read_jsonl(_published_results_path(Path(baseline_run)))
    _validate_publishable(baseline_manifest, baseline_results)
    _validate_pairing(manifest, results, baseline_manifest, baseline_results)
    baseline_by_id = {row["id"]: row for row in baseline_results}
    candidate_by_id = {row["id"]: row for row in results}

    comparison_specs = {
        "final_context_recall": (
            lambda row: row["stage_metrics"]["final_context_parent"]["recall_at_k"],
            -0.02,
        ),
        "final_context_ndcg": (
            lambda row: row["stage_metrics"]["final_context_parent"]["ndcg_at_k"],
            -0.02,
        ),
        "faithfulness": (
            lambda row: row.get("ragas_metrics", {}).get("faithfulness"),
            -0.02,
        ),
        "answer_correctness": (
            lambda row: row.get("ragas_metrics", {}).get("answer_correctness"),
            -0.02,
        ),
        "abstain_accuracy": (
            lambda row: row.get("behavior_metrics", {}).get("abstain_accuracy"),
            0.0,
        ),
        "deny_accuracy": (
            lambda row: row.get("behavior_metrics", {}).get("deny_accuracy"),
            0.0,
        ),
    }
    comparisons: dict[str, Any] = {}
    comparison_checks: dict[str, bool] = {}
    ordered_ids = sorted(candidate_by_id)
    for name, (getter, threshold) in comparison_specs.items():
        pairs = [
            (getter(baseline_by_id[case_id]), getter(candidate_by_id[case_id]))
            for case_id in ordered_ids
        ]
        valid_pairs = [
            (float(left), float(right))
            for left, right in pairs
            if _is_finite(left) and _is_finite(right)
        ]
        if not valid_pairs:
            comparisons[name] = None
            comparison_checks[name] = True
            continue
        interval = paired_bootstrap_delta(
            [left for left, _ in valid_pairs],
            [right for _, right in valid_pairs],
        )
        comparisons[name] = {
            **asdict(interval),
            "count": len(valid_pairs),
            "minimum_delta": threshold,
        }
        comparison_checks[name] = interval.estimate >= threshold

    baseline_p95 = _percentile(
        _finite_values(row.get("latency_ms") for row in baseline_results),
        0.95,
    )
    candidate_p95 = _percentile(
        _finite_values(row.get("latency_ms") for row in results),
        0.95,
    )
    latency_ratio = (
        candidate_p95 / baseline_p95
        if baseline_p95 is not None and baseline_p95 > 0 and candidate_p95 is not None
        else None
    )
    comparisons["p95_latency_ratio"] = latency_ratio
    comparison_checks["p95_latency_ratio"] = (
        latency_ratio is not None and latency_ratio <= 1.20
    )
    checks = {**safety_checks, **comparison_checks}
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "comparisons": comparisons,
    }


def _validate_pairing(
    candidate_manifest: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    baseline_manifest: dict[str, Any],
    baseline_results: list[dict[str, Any]],
) -> None:
    candidate_fingerprint = candidate_manifest.get("fingerprint") or {}
    baseline_fingerprint = baseline_manifest.get("fingerprint") or {}
    for key in ("dataset_sha256", "corpus_version"):
        if candidate_fingerprint.get(key) != baseline_fingerprint.get(key):
            raise ValueError(f"Baseline 与 Candidate 的 {key} 不一致")
    if {row.get("id") for row in candidate_results} != {
        row.get("id") for row in baseline_results
    }:
        raise ValueError("Baseline 与 Candidate 必须包含相同 Case")


def _build_report(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    *,
    baseline_run: Path | None,
) -> str:
    global_metrics = summary["dimensions"]["global"]["all"]["metrics"]
    lines = [
        "# 01_RAG 生产级真实评测报告",
        "",
        f"- Split：{manifest.get('split', 'unknown')}",
        f"- 已完成 Case：{summary['total']}",
        f"- 退化样本：{summary['degraded_case_count']}",
        f"- 权限泄漏证据数：{summary['permission_leakage_total']}",
        "- 指标区间：95% CI（bootstrap）",
        "",
        "## 分阶段检索指标",
        "",
    ]
    for stage_name, metric_name, label in STAGE_HIGHLIGHTS:
        lines.append(
            f"- {label}：{_format_summary(global_metrics[f'{stage_name}.{metric_name}'])}"
        )
    lines.extend(["", "## 生成质量", ""])
    for metric_name in RAGAS_METRIC_NAMES:
        lines.append(
            f"- {RAGAS_LABELS[metric_name]}："
            f"{_format_summary(global_metrics[f'ragas.{metric_name}'])}"
        )
    lines.extend(["", "## 安全与行为", ""])
    lines.append(
        "- Abstain Accuracy："
        + _format_summary(global_metrics["behavior.abstain_accuracy"])
    )
    lines.append(
        "- Deny Accuracy："
        + _format_summary(global_metrics["behavior.deny_accuracy"])
    )
    gates = summary["release_gates"]
    lines.extend(["", "## 发布门禁", ""])
    if baseline_run is None:
        lines.append("- 状态：未提供 Baseline，仅完成安全与完整性检查，不作发布结论。")
    else:
        lines.append(f"- 状态：{gates['status']}")
    for name, passed in gates["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}：{name}")
    lines.extend(
        [
            "",
            "> 本报告不计算 RAG 总分；中间阶段指标用于诊断，最终上下文、生成质量、安全和延迟用于发布门禁。",
            "",
        ]
    )
    return "\n".join(lines)


def _format_summary(summary: dict[str, Any]) -> str:
    if summary["average"] is None:
        return "n/a（有效样本 0）"
    interval = summary["ci95"]
    return (
        f"{summary['average']:.3f}，95% CI "
        f"[{interval['lower']:.3f}, {interval['upper']:.3f}]，"
        f"有效样本 {summary['count']}"
    )


def _write_stage_metrics(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "id": result["id"],
            "split": result["split"],
            "stage": stage_name,
            **metrics,
        }
        for result in results
        for stage_name, metrics in (result.get("stage_metrics") or {}).items()
    ]
    _write_jsonl(path, rows)


def _write_ragas_results(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [
        {
            "id": result["id"],
            "split": result["split"],
            **(result.get("ragas_metrics") or {}),
        }
        for result in results
        if result.get("expected_behavior") == "answer"
    ]
    _write_jsonl(path, rows)


def _published_results_path(run_dir: Path) -> Path:
    published = run_dir / "case_results.jsonl"
    return published if published.exists() else run_dir / "case_results.partial.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须包含 JSON 对象")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _finite_values(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if _is_finite(value)]


def _is_finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
