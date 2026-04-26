"""LLM-as-judge：对 Agent 产出的报告进行三维度打分。

复用 DeepSeek pro（json_mode 已在 planner/reflector 验证稳定）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from config.llm import get_llm

JUDGE_SYSTEM = """你是一名严格但公正的 AI 研究报告评审。
请基于"原始问题、Planner 拆解的子问题、检索得到的 evidence 摘要、最终报告"四方面信息，
对报告给出 0-100 的三维度评分（覆盖度、准确性、引用质量）。

评分准则：
- coverage  : 报告是否覆盖了所有 sub_questions；缺失关键子问题大幅扣分
- accuracy  : 关键论断是否能在 evidence 中找到支撑；明显幻觉、与 evidence 冲突大幅扣分
- citation  : 引用编号 [^N] 是否对应 evidence；关键数字/事实是否有引用支撑
- overall   : coverage*0.4 + accuracy*0.3 + citation*0.3，四舍五入到整数
- rationale : 中文一段话，每个维度给 1 句解释 + 主要失分点

只输出 JSON，不要解释，不要 markdown 代码块。"""


class JudgeScore(BaseModel):
    coverage: int = Field(ge=0, le=100)
    accuracy: int = Field(ge=0, le=100)
    citation: int = Field(ge=0, le=100)
    overall: int = Field(ge=0, le=100)
    rationale: str


@dataclass
class JudgeInput:
    query: str
    plan: list[dict]            # [{id, question, recommended_sources}]
    evidence_brief: list[dict]  # [{idx, source_type, source_url, snippet}]
    report_md: str


def _format_plan(plan: list[dict]) -> str:
    if not plan:
        return "(无 plan)"
    lines = []
    for sq in plan:
        srcs = ",".join(sq.get("recommended_sources", []) or [])
        lines.append(f"- {sq.get('id','?')}: {sq.get('question','')} [{srcs}]")
    return "\n".join(lines)


def _format_evidence(evidence: list[dict], limit: int = 30) -> str:
    if not evidence:
        return "(无 evidence)"
    out = []
    for i, ev in enumerate(evidence[:limit], 1):
        snip = (ev.get("snippet") or "").strip().replace("\n", " ")[:200]
        out.append(f"[{i}] ({ev.get('source_type','?')}) {ev.get('source_url','')}\n    {snip}")
    if len(evidence) > limit:
        out.append(f"... 还有 {len(evidence) - limit} 条 evidence 已省略")
    return "\n".join(out)


def build_judge_prompt(inp: JudgeInput, *, report_max_chars: int = 6000) -> str:
    report = inp.report_md
    if len(report) > report_max_chars:
        report = report[:report_max_chars] + f"\n\n...(报告已截断，共 {len(inp.report_md)} 字)"
    return (
        f"# 原始研究问题\n{inp.query}\n\n"
        f"# Planner 子问题\n{_format_plan(inp.plan)}\n\n"
        f"# Evidence 摘要\n{_format_evidence(inp.evidence_brief)}\n\n"
        f"# 报告全文\n{report}\n"
    )


async def judge_one(inp: JudgeInput, *, llm: Any | None = None) -> JudgeScore:
    """对单条评测样本打分。llm 可注入用于测试。"""
    judge_llm = llm if llm is not None else get_llm("max", temperature=0.0)
    structured = judge_llm.with_structured_output(JudgeScore, method="json_mode")
    score: JudgeScore = await structured.ainvoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=build_judge_prompt(inp))]
    )
    return score
