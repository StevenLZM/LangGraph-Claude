from __future__ import annotations

from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent.events import AgentEvent, AgentRunResult
from agent.prompts import build_plan_system_prompt
from agent.react import EventCallback, run_react
from config.llm import get_llm
from config.settings import settings
from tools.registry import get_tools


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
    react_events: list[AgentEvent] = Field(default_factory=list)


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
    tool_list = get_tools()
    structured = llm.with_structured_output(Plan, method="json_mode")
    result = structured.invoke(
        [
            SystemMessage(content=build_plan_system_prompt(tool_list)),
            HumanMessage(content=f"用户任务：{user_input}\n请输出 JSON。"),
        ]
    )
    if isinstance(result, Plan):
        return result
    return Plan.model_validate(result)


def _step_prompt(step: PlanStep, user_input: str, past_steps: list[StepResult]) -> str:
    history = "\n".join(f"{item.step_id}. {item.objective}: {item.output}" for item in past_steps) or "无"
    return (
        f"原始任务：{user_input}\n"
        f"已完成步骤：\n{history}\n\n"
        f"当前步骤：{step.objective}\n"
        f"建议工具：{step.suggested_tool}\n"
        "请完成当前步骤；如果需要工具，自动调用合适工具。"
    )


def default_executor(
    step: PlanStep,
    user_input: str,
    past_steps: list[StepResult],
    *,
    event_callback: EventCallback | None = None,
) -> StepResult:
    prompt = _step_prompt(step, user_input, past_steps)
    result = run_react(prompt, llm=get_llm("turbo", temperature=0.1), event_callback=event_callback)
    return StepResult(
        step_id=step.id,
        objective=step.objective,
        output=result.final_answer,
        tool_used=step.suggested_tool,
        react_events=result.events,
    )


def _final_answer(input_text: str, past_steps: list[StepResult]) -> str:
    if not past_steps:
        return f"未生成可执行步骤，无法完成任务：{input_text}"
    lines = [f"{item.step_id}. {item.objective}: {item.output}" for item in past_steps]
    return "执行完成：\n" + "\n".join(lines) + f"\n\n最终结论：{past_steps[-1].output}"


def _plan_event(plan: Plan) -> AgentEvent:
    plan_text = "\n".join(f"{step.id}. {step.objective} [{step.suggested_tool}]" for step in plan.steps)
    return AgentEvent(type="plan", title="执行计划", content=plan_text, metadata={"steps": plan.model_dump()})


def _step_event(step: StepResult) -> AgentEvent:
    return AgentEvent(
        type="step",
        title=f"步骤 {step.step_id}: {step.objective}",
        content=step.output,
        tool=step.tool_used,
        metadata=step.model_dump(),
    )


def _react_round_for_event(event: AgentEvent, current_round: int) -> tuple[int, int]:
    if event.type == "tool_call":
        next_round = current_round + 1
        return next_round, next_round
    if event.type == "final":
        return current_round, current_round + 1 if current_round else 1
    return current_round, current_round if current_round else 1


def _react_trace_event(step_id: int, objective: str, event: AgentEvent, react_round: int) -> AgentEvent:
    return AgentEvent(
        type=event.type,
        title=f"步骤 {step_id} / ReAct 第 {react_round} 轮 / {event.title}",
        content=event.content,
        tool=event.tool,
        tool_input=event.tool_input,
        tool_output=event.tool_output,
        metadata={
            **event.metadata,
            "parent_step_id": step_id,
            "parent_step_objective": objective,
            "react_round": react_round,
        },
    )


def _react_trace_events_for_step(step: StepResult) -> list[AgentEvent]:
    trace_events: list[AgentEvent] = []
    current_round = 0
    for event in step.react_events:
        current_round, react_round = _react_round_for_event(event, current_round)
        trace_events.append(_react_trace_event(step.step_id, step.objective, event, react_round))
    return trace_events


def _emit_event(events: list[AgentEvent], event: AgentEvent, event_callback: EventCallback | None) -> None:
    events.append(event)
    if event_callback is not None:
        event_callback(event)


def _run_plan_and_execute_with_event_callback(
    user_input: str,
    *,
    planner: Planner | None = None,
    executor: Executor | None = None,
    max_steps: int | None = None,
    event_callback: EventCallback,
) -> AgentRunResult:
    plan_fn = planner or default_planner
    limit = max_steps or settings.max_plan_steps
    plan = plan_fn(user_input)
    if len(plan.steps) > limit:
        plan = Plan(steps=plan.steps[:limit])

    events: list[AgentEvent] = []
    past_steps: list[StepResult] = []
    _emit_event(events, _plan_event(plan), event_callback)

    for step in plan.steps:
        if executor is None:
            current_round = 0

            def emit_react_trace(event: AgentEvent) -> None:
                nonlocal current_round
                current_round, react_round = _react_round_for_event(event, current_round)
                _emit_event(
                    events,
                    _react_trace_event(step.id, step.objective, event, react_round),
                    event_callback,
                )

            result = default_executor(step, user_input, past_steps, event_callback=emit_react_trace)
        else:
            result = executor(step, user_input, past_steps)
            for trace_event in _react_trace_events_for_step(result):
                _emit_event(events, trace_event, event_callback)

        past_steps.append(result)
        _emit_event(events, _step_event(result), event_callback)

    final = _final_answer(user_input, past_steps)
    _emit_event(events, AgentEvent(type="final", title="最终答案", content=final), event_callback)
    return AgentRunResult(
        final_answer=final,
        events=events,
        raw_state={"input": user_input, "plan": plan, "past_steps": past_steps, "final_answer": final},
    )


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
    event_callback: EventCallback | None = None,
) -> AgentRunResult:
    if event_callback is not None:
        return _run_plan_and_execute_with_event_callback(
            user_input,
            planner=planner,
            executor=executor,
            max_steps=max_steps,
            event_callback=event_callback,
        )

    graph = build_plan_execute_graph(planner=planner, executor=executor, max_steps=max_steps)
    state = graph.invoke({"input": user_input})

    events: list[AgentEvent] = []
    plan = state.get("plan", Plan(steps=[]))
    events.append(_plan_event(plan))
    for step in state.get("past_steps", []):
        events.append(_step_event(step))
        events.extend(_react_trace_events_for_step(step))
    final = state.get("final_answer", "")
    events.append(AgentEvent(type="final", title="最终答案", content=final))
    return AgentRunResult(final_answer=final, events=events, raw_state=dict(state))
