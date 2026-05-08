from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def build_report(results_path: Path) -> str:
    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total = len(results)
    average = sum(float(item["score"]) for item in results) / total if total else 0.0
    passed = sum(1 for item in results if item.get("passed"))
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in results:
        by_category[item["category"]].append(float(item["score"]))

    lines = [
        "# M6 自动评测报告",
        "",
        f"- 用例数：{total}",
        f"- 通过数：{passed}",
        f"- 平均分：{average:.1f}",
        "",
        "## 分类得分",
        "",
    ]
    for category in sorted(by_category):
        scores = by_category[category]
        lines.append(f"- {category}: {sum(scores) / len(scores):.1f}")
    return "\n".join(lines) + "\n"
