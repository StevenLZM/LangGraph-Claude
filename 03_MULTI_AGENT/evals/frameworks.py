"""Teaching-oriented eval framework coverage for InsightLoop.

This module does not pretend every production metric is fully available from a
single offline eval record. It makes coverage explicit so students can see what
is automated, what is proxy-only, and what needs trace export or human review.
"""
from __future__ import annotations

from typing import Any


EVAL_FRAMEWORKS: list[dict[str, str]] = [
    {
        "id": "pytest_contracts",
        "name": "Pytest contract/unit tests",
        "layer": "testing",
        "teaching_goal": "保护 reducer、registry、runtime skill、writer citation 等确定性契约",
    },
    {
        "id": "graph_e2e",
        "name": "Offline graph E2E",
        "layer": "testing",
        "teaching_goal": "用打桩 LLM 和 registry 验证 planner interrupt 到 writer 的闭环",
    },
    {
        "id": "golden_dataset",
        "name": "Golden dataset regression",
        "layer": "result",
        "teaching_goal": "固定 20 题数据集做版本回归和类别覆盖",
    },
    {
        "id": "llm_as_judge",
        "name": "LLM-as-judge",
        "layer": "result",
        "teaching_goal": "评估最终报告 coverage / accuracy / citation / overall",
    },
    {
        "id": "deterministic_guardrails",
        "name": "Deterministic guardrails",
        "layer": "process",
        "teaching_goal": "用代码检查 citation audit、空 evidence、force_complete 等硬信号",
    },
    {
        "id": "retrieval_proxy",
        "name": "RAG/Retrieval proxy",
        "layer": "process",
        "teaching_goal": "用 subquestion recall、source type recall、evidence density 近似检索覆盖",
    },
    {
        "id": "component_quality",
        "name": "Component quality",
        "layer": "process",
        "teaching_goal": "把 Planner、Research、Reflector、Writer、Runtime 分开诊断",
    },
    {
        "id": "runtime_reliability",
        "name": "Runtime reliability",
        "layer": "runtime",
        "teaching_goal": "观察 error、elapsed_sec、slow_case、error_type",
    },
    {
        "id": "no_feedback_sampling",
        "name": "No-feedback sampling",
        "layer": "sampling",
        "teaching_goal": "按风险原因决定 manual_review / spot_check / random_sample",
    },
    {
        "id": "human_review_rubric",
        "name": "Human review rubric",
        "layer": "sampling",
        "teaching_goal": "提供 pass / minor / major / unsafe / unjudgeable 的人工复核模板",
    },
    {
        "id": "trace_observability",
        "name": "Trace observability",
        "layer": "runtime",
        "teaching_goal": "说明 LangSmith trace 或节点级指标如何补齐生产诊断",
    },
    {
        "id": "comparative_ab",
        "name": "Run-to-run / A-B comparison",
        "layer": "analysis",
        "teaching_goal": "通过两次 run 的 score delta 观察版本变化",
    },
]


HUMAN_REVIEW_LABELS = {
    "pass": "可直接交付",
    "minor_issue": "小缺陷，不影响主要结论",
    "major_issue": "遗漏、错误引用、证据不足或关键论证弱",
    "unsafe_or_misleading": "明显误导、编造、关键事实错",
    "unjudgeable": "证据不足，无法判断",
}


def _has(record: dict[str, Any], key: str) -> bool:
    value = record.get(key)
    return value is not None and value != {}


def _status_for(framework_id: str, record: dict[str, Any]) -> str:
    if framework_id in {"pytest_contracts", "graph_e2e"}:
        return "covered_by_pytest"
    if framework_id == "golden_dataset":
        return "covered" if _has(record, "case") else "missing"
    if framework_id == "llm_as_judge":
        return "covered" if _has(record, "score") else "missing"
    if framework_id == "deterministic_guardrails":
        return "covered" if _has(record, "process_quality") else "missing"
    if framework_id == "retrieval_proxy":
        return "covered" if _has(record, "retrieval_metrics") else "missing"
    if framework_id == "component_quality":
        return "covered" if _has(record, "component_quality") else "missing"
    if framework_id == "runtime_reliability":
        return "covered" if _has(record, "runtime_health") else "missing"
    if framework_id == "no_feedback_sampling":
        return "covered" if _has(record, "sampling") else "missing"
    if framework_id == "human_review_rubric":
        return "covered" if _has(record, "human_review") else "template"
    if framework_id == "trace_observability":
        return "covered" if _has(record, "node_metrics") or _has(record, "trace_url") else "partial"
    if framework_id == "comparative_ab":
        return "report_level"
    return "unknown"


def build_framework_coverage(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return framework coverage rows for a single eval record."""
    rows = []
    for framework in EVAL_FRAMEWORKS:
        rows.append({
            **framework,
            "status": _status_for(framework["id"], record),
        })
    return rows


def framework_matrix_markdown(rows: list[dict[str, str]]) -> str:
    """Render the teaching framework coverage matrix."""
    table = [
        "## 教学评估框架覆盖矩阵",
        "",
        "| 框架 | 层级 | 状态 | 教学目标 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        table.append(
            f"| {row['name']} | {row['layer']} | {row['status']} | {row['teaching_goal']} |"
        )
    table.append("")
    table.append("**人工复核标签**：" + "；".join(f"`{k}`={v}" for k, v in HUMAN_REVIEW_LABELS.items()))
    return "\n".join(table)
