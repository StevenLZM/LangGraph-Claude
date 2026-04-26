"""evals/report.py —— 把 results.jsonl 渲染成 Markdown 评测报告。"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def _load_results(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _safe_avg(values: Iterable[float]) -> float | None:
    vs = [v for v in values if v is not None]
    if not vs:
        return None
    return round(mean(vs), 1)


def _render_table(records: list[dict]) -> str:
    head = "| 用例 | 类别 | 覆盖 | 准确 | 引用 | 总分 | 用时(s) | 状态 | 报告 |"
    sep  = "|---|---|---:|---:|---:|---:|---:|---|---|"
    rows = [head, sep]
    for rec in records:
        case = rec["case"]
        score = rec.get("score") or {}
        err = rec.get("error")
        if err:
            rows.append(
                f"| {case['id']} | {case.get('category','')} | - | - | - | - | "
                f"{rec.get('elapsed_sec','-')} | ❌ {err[:40]} | - |"
            )
            continue
        report_link = rec.get("report_path") or "-"
        if report_link != "-":
            report_link = f"[{Path(report_link).name}]({report_link})"
        rows.append(
            f"| {case['id']} | {case.get('category','')} "
            f"| {score.get('coverage','-')} | {score.get('accuracy','-')} | {score.get('citation','-')} "
            f"| **{score.get('overall','-')}** | {rec.get('elapsed_sec','-')} | ✅ | {report_link} |"
        )
    return "\n".join(rows)


def _render_means(records: list[dict]) -> str:
    scores = [r.get("score") for r in records if r.get("score") and not r.get("error")]
    if not scores:
        return "**维度均值**：(无成功用例)"
    cov = _safe_avg(s["coverage"] for s in scores)
    acc = _safe_avg(s["accuracy"] for s in scores)
    cit = _safe_avg(s["citation"] for s in scores)
    overall = _safe_avg(s["overall"] for s in scores)
    return (
        "## 维度均值\n"
        f"- 覆盖度: **{cov}**\n"
        f"- 准确性: **{acc}**\n"
        f"- 引用质量: **{cit}**\n"
        f"- 综合: **{overall}**"
    )


def _render_lowlights(records: list[dict], threshold: int = 70) -> str:
    bad = [r for r in records if (r.get("score") or {}).get("overall", 100) < threshold and not r.get("error")]
    err = [r for r in records if r.get("error")]
    if not bad and not err:
        return "## 失分案例\n(全部通过 70 分线)"
    parts = ["## 失分案例"]
    for r in bad:
        s = r["score"]
        parts.append(
            f"### {r['case']['id']}（综合 {s['overall']}）\n"
            f"- 覆盖 {s['coverage']} / 准确 {s['accuracy']} / 引用 {s['citation']}\n"
            f"- {s.get('rationale','')}"
        )
    for r in err:
        parts.append(f"### {r['case']['id']} —— ❌ 执行失败\n- 错误: `{r['error']}`")
    return "\n\n".join(parts)


def render_markdown(results_path: Path, out_path: Path, *, run_id: str) -> Path:
    records = _load_results(results_path)
    body = (
        f"# InsightLoop Eval Report — `{run_id}`\n\n"
        f"- 用例总数: **{len(records)}**\n"
        f"- 成功: **{sum(1 for r in records if not r.get('error'))}**\n"
        f"- 数据来源: `{results_path}`\n\n"
        f"## 明细\n\n{_render_table(records)}\n\n"
        f"{_render_means(records)}\n\n"
        f"{_render_lowlights(records)}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
