from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.ragas_adapter import RAGAS_METRIC_NAMES


METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_correctness": "Answer Correctness",
    "answer_relevancy": "Answer Relevancy",
    "semantic_similarity": "Semantic Similarity",
}


def build_report(
    results_path: Path,
    *,
    metric_names: list[str] | None = None,
    summary_path: Path | None = None,
) -> str:
    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = metric_names or RAGAS_METRIC_NAMES
    summary = summarize_results(results, metrics)
    if summary_path is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    lines = [
        "# 01_RAG RAGAS 评估报告",
        "",
        f"- 用例数：{summary['total']}",
        "- 评估框架：RAGAS",
        "",
        "## RAGAS 指标",
        "",
    ]
    for metric_name in metrics:
        item = summary["metrics"][metric_name]
        lines.append(
            f"- {METRIC_LABELS.get(metric_name, metric_name)}："
            f"平均 {item['average']:.3f}，有效样本 {item['count']}"
        )

    lines.extend(["", "## 分类指标", ""])
    for category in sorted(summary["by_category"]):
        item = summary["by_category"][category]
        metric_parts = [
            f"{METRIC_LABELS.get(name, name)}={item['metrics'][name]['average']:.3f}"
            for name in metrics
        ]
        lines.append(f"- {category}: cases={item['total']}, " + ", ".join(metric_parts))

    lines.extend(["", "## 样本明细", ""])
    for result in results:
        metric_parts = [
            f"{METRIC_LABELS.get(name, name)}={_fmt_metric(result.get(name))}"
            for name in metrics
        ]
        lines.append(
            f"- `{result.get('id', '')}` [{result.get('category', 'unknown')}] "
            + ", ".join(metric_parts)
        )
    return "\n".join(lines) + "\n"


def summarize_results(
    results: list[dict[str, Any]],
    metric_names: list[str] | None = None,
) -> dict[str, Any]:
    metrics = metric_names or RAGAS_METRIC_NAMES
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[str(result.get("category", "unknown"))].append(result)

    return {
        "total": len(results),
        "metrics": {
            metric_name: _metric_summary(results, metric_name)
            for metric_name in metrics
        },
        "by_category": {
            category: {
                "total": len(items),
                "metrics": {
                    metric_name: _metric_summary(items, metric_name)
                    for metric_name in metrics
                },
            }
            for category, items in by_category.items()
        },
    }


def _metric_summary(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(item[key])
        for item in items
        if item.get(key) is not None and _is_number(item[key])
    ]
    if not values:
        return {"average": 0.0, "count": 0}
    return {
        "average": sum(values) / len(values),
        "count": len(values),
    }


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _fmt_metric(value: Any) -> str:
    if value is None or not _is_number(value):
        return "n/a"
    return f"{float(value):.3f}"
