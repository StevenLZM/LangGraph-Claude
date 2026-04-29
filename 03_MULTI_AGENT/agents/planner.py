"""Planner —— async LLM + interrupt HITL。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from agents.schemas import ResearchPlan
from config.llm import get_llm
from graph.state import ResearchState
from prompts.templates import PLANNER_SYSTEM, planner_user

logger = logging.getLogger(__name__)


async def planner_node(state: ResearchState) -> dict[str, Any]:
    query = state.get("research_query", "")
    audience = state.get("audience", "intermediate")

    llm = get_llm("max", temperature=0.3)
    structured = llm.with_structured_output(ResearchPlan, method="json_mode")
    plan: ResearchPlan = await structured.ainvoke(
        [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=planner_user(query, audience))]
    )
    logger.info("[planner] 生成 %d 个子问题", len(plan.sub_questions))
    print(f"llm生成plan：{plan}")

    # model_dump() 是 Pydantic v2 的方法，将模型实例序列化为字典。
    # plan = Plan(name="数据分析", steps=["采集", "清洗", "建模"])
    # result = plan.model_dump()
    # {'name': '数据分析', 'steps': ['采集', '清洗', '建模'], 'priority': 1}
    decision = interrupt({"phase": "plan_review", "plan": plan.model_dump()})
    print(f"用户的输入:{decision}")

    confirmed = _coerce_plan(decision, fallback=plan)
    print(f"转换后的输入:{confirmed}")
    return {
        "plan": confirmed.sub_questions,
        "plan_confirmed": True,
        "iteration": 0,
        "revision_count": 0,
        "current_node": "planner",
        "messages": [
            AIMessage(content=f"已生成 {len(confirmed.sub_questions)} 个子问题，进入并行调研阶段")
        ],
    }


def _coerce_plan(decision: Any, *, fallback: ResearchPlan) -> ResearchPlan:
    if decision is None:
        return fallback
    if isinstance(decision, ResearchPlan):
        return decision
    if isinstance(decision, dict):
        plan_payload = decision.get("plan", decision)
        if isinstance(plan_payload, ResearchPlan):
            return plan_payload
        if isinstance(plan_payload, dict) and "sub_questions" in plan_payload:
            try:
                return ResearchPlan.model_validate(plan_payload)
            except Exception as e:
                logger.warning("[planner] resume 数据解析失败，回退默认 plan: %s", e)
    return fallback
