from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_report(results_path: Path, summary_path: Path | None = None) -> str:
    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_results(results)
    if summary_path is not None:
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    lines = [
        "# 01_RAG 检索效果评测报告",
        "",
        f"- 用例数：{summary['total']}",
        f"- 通过数：{summary['passed']}",
        f"- 平均检索分：{summary['average_retrieval_score']:.1f}",
        f"- Recall@5：{summary['average_recall@5']:.3f}",
        f"- MRR：{summary['average_mrr']:.3f}",
        f"- Parent Hit Rate：{summary['parent_hit_rate']:.3f}",
        f"- Time Intent Accuracy：{summary['time_intent_accuracy']:.3f}",
        f"- Time Filter Accuracy：{summary['time_filter_accuracy']:.3f}",
        "",
        "## 分类指标",
        "",
    ]
    for category in sorted(summary["by_category"]):
        item = summary["by_category"][category]
        lines.append(
            f"- {category}: score={item['average_retrieval_score']:.1f}, "
            f"Recall@5={item['average_recall@5']:.3f}, MRR={item['average_mrr']:.3f}, "
            f"ParentHit={item['parent_hit_rate']:.3f}, cases={item['total']}"
        )

    generation_total = summary.get("generation_total", 0)
    if generation_total:
        lines.extend(
            [
                "",
                "## 生成质量",
                "",
                f"- 生成用例数：{generation_total}",
                f"- 平均生成分：{summary['average_generation_score']:.1f}",
            ]
        )
    return "\n".join(lines) + "\n"


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[str(result.get("category", "unknown"))].append(result)

    generation_results = [item for item in results if "generation_score" in item]
    return {
        "total": total,
        "passed": sum(1 for item in results if item.get("passed")),
        "average_retrieval_score": _avg(results, "retrieval_score"),
        "average_recall@5": _avg(results, "recall@5"),
        "average_mrr": _avg(results, "mrr"),
        "parent_hit_rate": _rate(results, "parent_hit"),
        "time_intent_accuracy": _avg_defined(results, "time_intent_accuracy"),
        "time_filter_accuracy": _avg_defined(results, "time_filter_accuracy"),
        "generation_total": len(generation_results),
        "average_generation_score": _avg(generation_results, "generation_score"),
        "by_category": {
            category: {
                "total": len(items),
                "average_retrieval_score": _avg(items, "retrieval_score"),
                "average_recall@5": _avg(items, "recall@5"),
                "average_mrr": _avg(items, "mrr"),
                "parent_hit_rate": _rate(items, "parent_hit"),
                "time_intent_accuracy": _avg_defined(items, "time_intent_accuracy"),
                "time_filter_accuracy": _avg_defined(items, "time_filter_accuracy"),
            }
            for category, items in by_category.items()
        },
    }


def _avg(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _avg_defined(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rate(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get(key)) / len(items)
