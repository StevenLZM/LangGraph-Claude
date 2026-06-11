"""evals/report.py —— 把 results.jsonl 渲染成 Markdown 评测报告。"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from evals.frameworks import build_framework_coverage, framework_matrix_markdown
from evals.manual_review import build_review_queue


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


def _process_quality(rec: dict) -> dict:
    if rec.get("process_quality"):
        return rec["process_quality"]
    return {
        "plan_size": rec.get("plan_size"),
        "evidence_count": rec.get("evidence_count"),
        "source_diversity": None,
        "missing_subquestions": [],
        "forced_completion": False,
        "citation_audit_issue_count": 0,
    }


def _runtime_health(rec: dict) -> dict:
    if rec.get("runtime_health"):
        return rec["runtime_health"]
    return {
        "status": "error" if rec.get("error") else "ok",
        "elapsed_sec": rec.get("elapsed_sec"),
        "slow_case": False,
        "error_type": (rec.get("error") or "").split(":", 1)[0] or None,
    }


def _retrieval_metrics(rec: dict) -> dict:
    return rec.get("retrieval_metrics") or {
        "subquestion_recall": None,
        "source_type_recall": None,
        "evidence_density": None,
    }


def _component_quality(rec: dict) -> dict:
    return rec.get("component_quality") or {}


def _node_metrics(rec: dict) -> dict:
    return rec.get("node_metrics") or {}


def _render_three_layer_coverage(records: list[dict]) -> str:
    total = len(records)
    result_quality = sum(1 for r in records if r.get("score") and not r.get("error"))
    process_quality = sum(1 for r in records if r.get("process_quality") is not None)
    runtime_health = sum(1 for r in records if r.get("runtime_health") is not None)
    return (
        "## 三层评估覆盖\n"
        f"- 结果质量: **{result_quality}/{total}**（LLM-as-judge coverage / accuracy / citation）\n"
        f"- 过程链路质量: **{process_quality}/{total}**（plan / evidence / reflection / citation audit）\n"
        f"- 运行稳定性: **{runtime_health}/{total}**（error / elapsed / slow case）"
    )


def _render_process_quality(records: list[dict]) -> str:
    pqs = [_process_quality(r) for r in records]
    avg_plan = _safe_avg(pq.get("plan_size") for pq in pqs)
    avg_evidence = _safe_avg(pq.get("evidence_count") for pq in pqs)
    avg_sources = _safe_avg(pq.get("source_diversity") for pq in pqs)
    missing_cnt = sum(1 for pq in pqs if pq.get("missing_subquestions"))
    forced_cnt = sum(1 for pq in pqs if pq.get("forced_completion"))
    citation_issue_cnt = sum(1 for pq in pqs if pq.get("citation_audit_issue_count", 0) > 0)
    return (
        "## 过程链路质量\n"
        f"- 平均 plan size: **{avg_plan}**\n"
        f"- 平均 evidence count: **{avg_evidence}**\n"
        f"- 平均来源多样性: **{avg_sources}**\n"
        f"- 子问题无 evidence 用例: **{missing_cnt}**\n"
        f"- force_complete 用例: **{forced_cnt}**\n"
        f"- citation audit 告警用例: **{citation_issue_cnt}**"
    )


def _render_retrieval_metrics(records: list[dict]) -> str:
    metrics = [_retrieval_metrics(r) for r in records]
    subq_recall = _safe_avg(m.get("subquestion_recall") for m in metrics)
    source_recall = _safe_avg(m.get("source_type_recall") for m in metrics)
    density = _safe_avg(m.get("evidence_density") for m in metrics)
    return (
        "## RAG/Retrieval Proxy\n"
        f"- 平均 subquestion recall: **{subq_recall}**\n"
        f"- 平均 source type recall: **{source_recall}**\n"
        f"- 平均 evidence density: **{density}**"
    )


def _render_component_quality(records: list[dict]) -> str:
    rows = ["| 组件 | 平均分 | fail/warn/pass |", "|---|---:|---|"]
    for name in ("planner", "research", "reflector", "writer", "runtime"):
        entries = [
            _component_quality(r).get(name)
            for r in records
            if _component_quality(r).get(name)
        ]
        avg = _safe_avg(e.get("score") for e in entries)
        status_counts = {"fail": 0, "warn": 0, "pass": 0}
        for entry in entries:
            status = entry.get("status")
            if status in status_counts:
                status_counts[status] += 1
        rows.append(
            f"| {name} | {avg} | "
            f"{status_counts['fail']}/{status_counts['warn']}/{status_counts['pass']} |"
        )
    return "## 组件质量评分\n\n" + "\n".join(rows)


def _render_runtime_health(records: list[dict]) -> str:
    rhs = [_runtime_health(r) for r in records]
    errors = [rh for rh in rhs if rh.get("status") == "error"]
    slow_cnt = sum(1 for rh in rhs if rh.get("slow_case"))
    avg_elapsed = _safe_avg(rh.get("elapsed_sec") for rh in rhs)
    error_types = {}
    for rh in errors:
        et = rh.get("error_type") or "Error"
        error_types[et] = error_types.get(et, 0) + 1
    err_text = ", ".join(f"{k}={v}" for k, v in sorted(error_types.items())) or "无"
    return (
        "## 运行稳定性\n"
        f"- 执行失败: **{len(errors)}**\n"
        f"- 慢用例: **{slow_cnt}**\n"
        f"- 平均用时(s): **{avg_elapsed}**\n"
        f"- 错误类型: {err_text}"
    )


def _render_trace_metrics(records: list[dict]) -> str:
    metrics = [_node_metrics(r) for r in records]
    llm_calls = sum(int(m.get("llm_calls") or 0) for m in metrics)
    llm_errors = sum(int(m.get("llm_errors") or 0) for m in metrics)
    tool_calls = sum(int(m.get("tool_calls") or 0) for m in metrics)
    tool_errors = sum(int(m.get("tool_errors") or 0) for m in metrics)
    total_tokens = sum(int(m.get("total_tokens") or 0) for m in metrics)
    total_elapsed_ms = sum(float(m.get("total_elapsed_ms") or 0.0) for m in metrics)
    return (
        "## Trace Metrics\n"
        f"- LLM calls/errors: **{llm_calls}/{llm_errors}**\n"
        f"- Tool calls/errors: **{tool_calls}/{tool_errors}**\n"
        f"- Total tokens: **{total_tokens}**\n"
        f"- Callback elapsed(ms): **{round(total_elapsed_ms, 1)}**"
    )


def _render_sampling(records: list[dict]) -> str:
    rows = ["| 用例 | 风险 | 建议 | 原因 |", "|---|---|---|---|"]
    for rec in records:
        sampling = rec.get("sampling") or {}
        reasons = ", ".join(sampling.get("reasons") or []) or "-"
        rows.append(
            f"| {rec['case']['id']} | {sampling.get('risk_level', '-')} "
            f"| {sampling.get('review_action', '-')} | {reasons} |"
        )
    return "## 抽检建议\n\n" + "\n".join(rows)


def _render_manual_review_queue(records: list[dict]) -> str:
    queue = build_review_queue(records)
    rows = ["| 用例 | 风险 | 建议标签 | 原因 |", "|---|---|---|---|"]
    for item in queue:
        rows.append(
            f"| {item.get('case_id')} | {item.get('risk_level')} | "
            f"{item.get('suggested_label')} | {', '.join(item.get('reasons') or []) or '-'} |"
        )
    if not queue:
        rows.append("| - | - | - | - |")
    return "## 人工复核队列\n\n" + "\n".join(rows)


def _render_framework_matrix(records: list[dict]) -> str:
    if not records:
        return "## 教学评估框架覆盖矩阵\n\n(无记录)"
    rows = records[0].get("framework_coverage") or build_framework_coverage(records[0])
    return framework_matrix_markdown(rows)


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
        f"{_render_three_layer_coverage(records)}\n\n"
        f"{_render_process_quality(records)}\n\n"
        f"{_render_retrieval_metrics(records)}\n\n"
        f"{_render_component_quality(records)}\n\n"
        f"{_render_runtime_health(records)}\n\n"
        f"{_render_trace_metrics(records)}\n\n"
        f"{_render_sampling(records)}\n\n"
        f"{_render_manual_review_queue(records)}\n\n"
        f"{_render_framework_matrix(records)}\n\n"
        f"{_render_lowlights(records)}\n"
    )
    out_path.write_text(body, encoding="utf-8")
    return out_path
