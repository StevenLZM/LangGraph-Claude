from __future__ import annotations

from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.events import AgentEvent, AgentRunResult
from agent.prompts import PLAN_SYSTEM_PROMPT
from agent.react import run_react
from config.llm import get_llm
from config.settings import settings


class PlanStep(BaseModel):
    id: int = Field(ge=1)
    objective: str
    suggested_tool: str = "none"


class Plan(BaseModel):
    steps: list[PlanStep]


class StepResult(BaseModel):
    step_id: int
    objective: str
    output: str
    tool_used: str = "none"


class PlanExecuteState(TypedDict, total=False):
    input: str
    plan: Plan
    current_step: int
    past_steps: list[StepResult]
    final_answer: str


Planner = Callable[[str], Plan]
Executor = Callable[[PlanStep, str, list[StepResult]], StepResult]


def default_planner(user_input: str) -> Plan:
    llm = get_llm("max", temperature=0.1)
    structured = llm.with_structured_output(Plan, method="json_mode")
    result = structured.invoke(
        [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=f"用户任务：{user_input}\n请输出 JSON。"),
        ]
    )
    if isinstance(result, Plan):
        return result
    return Plan.model_validate(result)


def default_executor(step: PlanStep, user_input: str, past_steps: list[StepResult]) -> StepResult:
    history = "\n".join(f"{item.step_id}. {item.objective}: {item.output}" for item in past_steps) or "无"
    prompt = (
        f"原始任务：{user_input}\n"
        f"已完成步骤：\n{history}\n\n"
        f"当前步骤：{step.objective}\n"
        f"建议工具：{step.suggested_tool}\n"
        "请完成当前步骤；如果需要工具，自动调用合适工具。"
    )
    result = run_react(prompt, llm=get_llm("turbo", temperature=0.1))
    return StepResult(
        step_id=step.id,
        objective=step.objective,
        output=result.final_answer,
        tool_used=step.suggested_tool,
    )


def _final_answer(input_text: str, past_steps: list[StepResult]) -> str:
    if not past_steps:
        return f"未生成可执行步骤，无法完成任务：{input_text}"
    lines = [f"{item.step_id}. {item.objective}: {item.output}" for item in past_steps]
    return "执行完成：\n" + "\n".join(lines) + f"\n\n最终结论：{past_steps[-1].output}"


def build_plan_execute_graph(
    *,
    planner: Planner | None = None,
    executor: Executor | None = None,
    max_steps: int | None = None,
):
    plan_fn = planner or default_planner
    execute_fn = executor or default_executor
    limit = max_steps or settings.max_plan_steps

    def planner_node(state: PlanExecuteState) -> PlanExecuteState:
        plan = plan_fn(state["input"])
        if len(plan.steps) > limit:
            plan = Plan(steps=plan.steps[:limit])
        return {"plan": plan, "current_step": 0, "past_steps": []}

    def route_after_plan(state: PlanExecuteState) -> str:
        return "executor" if state["plan"].steps else "finalizer"

    def executor_node(state: PlanExecuteState) -> PlanExecuteState:
        current = int(state.get("current_step", 0))
        step = state["plan"].steps[current]
        past = list(state.get("past_steps", []))
        result = execute_fn(step, state["input"], past)
        return {"past_steps": [*past, result], "current_step": current + 1}

    def route_after_execute(state: PlanExecuteState) -> str:
        return "executor" if int(state.get("current_step", 0)) < len(state["plan"].steps) else "finalizer"

    def finalizer_node(state: PlanExecuteState) -> PlanExecuteState:
        return {"final_answer": _final_answer(state["input"], state.get("past_steps", []))}

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("finalizer", finalizer_node)
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", route_after_plan, {"executor": "executor", "finalizer": "finalizer"})
    workflow.add_conditional_edges("executor", route_after_execute, {"executor": "executor", "finalizer": "finalizer"})
    workflow.add_edge("finalizer", END)
    return workflow.compile()


def run_plan_and_execute(
    user_input: str,
    *,
    planner: Planner | None = None,
    executor: Executor | None = None,
    max_steps: int | None = None,
) -> AgentRunResult:
    graph = build_plan_execute_graph(planner=planner, executor=executor, max_steps=max_steps)
    state = graph.invoke({"input": user_input})

    events: list[AgentEvent] = []
    plan = state.get("plan", Plan(steps=[]))
    plan_text = "\n".join(f"{step.id}. {step.objective} [{step.suggested_tool}]" for step in plan.steps)
    events.append(AgentEvent(type="plan", title="执行计划", content=plan_text, metadata={"steps": plan.model_dump()}))
    for step in state.get("past_steps", []):
        events.append(
            AgentEvent(
                type="step",
                title=f"步骤 {step.step_id}: {step.objective}",
                content=step.output,
                tool=step.tool_used,
                metadata=step.model_dump(),
            )
        )
    final = state.get("final_answer", "")
    events.append(AgentEvent(type="final", title="最终答案", content=final))
    return AgentRunResult(final_answer=final, events=events, raw_state=dict(state))
