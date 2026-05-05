from __future__ import annotations

from agent import plan_execute
from agent.events import AgentEvent, AgentRunResult
from agent.plan_execute import (
    Plan,
    PlanStep,
    StepResult,
    build_plan_execute_graph,
    default_executor,
    run_plan_and_execute,
)


def _fake_planner(user_input: str) -> Plan:
    assert "北京" in user_input
    return Plan(
        steps=[
            PlanStep(id=1, objective="查询北京天气", suggested_tool="weather_query"),
            PlanStep(id=2, objective="判断是否适合跑步", suggested_tool="none"),
        ]
    )


def _fake_executor(step: PlanStep, user_input: str, past_steps: list[StepResult]) -> StepResult:
    if step.id == 1:
        return StepResult(step_id=1, objective=step.objective, output="北京晴，18℃，微风", tool_used="weather_query")
    return StepResult(step_id=2, objective=step.objective, output="适合跑步", tool_used="none")


def test_plan_execute_graph_runs_plan_steps_to_final_answer():
    graph = build_plan_execute_graph(planner=_fake_planner, executor=_fake_executor)

    state = graph.invoke({"input": "今天北京天气怎么样，适合跑步吗？"})

    assert [step.objective for step in state["plan"].steps] == ["查询北京天气", "判断是否适合跑步"]
    assert len(state["past_steps"]) == 2
    assert "适合跑步" in state["final_answer"]


def test_run_plan_and_execute_returns_plan_step_and_final_events():
    result = run_plan_and_execute("今天北京天气怎么样，适合跑步吗？", planner=_fake_planner, executor=_fake_executor)

    event_types = [event.type for event in result.events]
    assert event_types[0] == "plan"
    assert event_types.count("step") == 2
    assert event_types[-1] == "final"
    assert "北京晴" in result.final_answer


def test_run_plan_and_execute_expands_nested_react_trace_events():
    def traced_executor(step: PlanStep, user_input: str, past_steps: list[StepResult]) -> StepResult:
        return StepResult(
            step_id=step.id,
            objective=step.objective,
            output="达到最大工具调用步数，未能生成最终答案。",
            tool_used=step.suggested_tool,
            react_events=[
                AgentEvent(
                    type="tool_call",
                    title="调用工具: get_datetime",
                    tool="get_datetime",
                    tool_input={"timezone": "Asia/Shanghai"},
                    metadata={"id": "datetime-1"},
                ),
                AgentEvent(
                    type="tool_result",
                    title="工具结果: get_datetime",
                    content="当前时间: 2026年05月04日 12:00:00",
                    tool="get_datetime",
                    tool_output="当前时间: 2026年05月04日 12:00:00",
                    metadata={"tool_call_id": "datetime-1"},
                ),
                AgentEvent(
                    type="error",
                    title="执行停止",
                    content="达到最大工具调用步数，未能生成最终答案。",
                ),
            ],
        )

    result = run_plan_and_execute("今天北京天气怎么样，适合跑步吗？", planner=_fake_planner, executor=traced_executor)

    trace_events = [
        event for event in result.events if event.metadata.get("parent_step_id") == 1
    ]
    assert [event.type for event in trace_events] == ["tool_call", "tool_result", "error"]
    assert trace_events[0].title == "步骤 1 / ReAct 第 1 轮 / 调用工具: get_datetime"
    assert trace_events[0].metadata["react_round"] == 1
    assert trace_events[1].title == "步骤 1 / ReAct 第 1 轮 / 工具结果: get_datetime"
    assert trace_events[1].metadata["react_round"] == 1
    assert trace_events[2].title == "步骤 1 / ReAct 第 1 轮 / 执行停止"
    assert trace_events[2].content == "达到最大工具调用步数，未能生成最终答案。"


def test_default_executor_preserves_nested_react_events(monkeypatch):
    nested_events = [
        AgentEvent(
            type="tool_call",
            title="调用工具: web_search",
            tool="web_search",
            tool_input={"query": "A股 最新 指数"},
        )
    ]

    def fake_get_llm(tier: str, *, temperature: float):
        assert tier == "turbo"
        assert temperature == 0.1
        return object()

    def fake_run_react(prompt: str, *, llm: object, event_callback=None):
        assert event_callback is None
        assert "原始任务：最新的A股指数" in prompt
        assert "当前步骤：查询最新A股指数" in prompt
        return AgentRunResult(final_answer="上证指数 3000 点", events=nested_events, raw_state={})

    monkeypatch.setattr(plan_execute, "get_llm", fake_get_llm)
    monkeypatch.setattr(plan_execute, "run_react", fake_run_react)

    result = default_executor(
        PlanStep(id=1, objective="查询最新A股指数", suggested_tool="web_search"),
        "最新的A股指数",
        [],
    )

    assert result.output == "上证指数 3000 点"
    assert result.react_events == nested_events


def test_run_plan_and_execute_streams_nested_react_trace_events(monkeypatch):
    def one_step_planner(user_input: str) -> Plan:
        assert user_input == "最新的A股指数"
        return Plan(steps=[PlanStep(id=1, objective="查询最新A股指数", suggested_tool="web_search")])

    nested_events = [
        AgentEvent(
            type="tool_call",
            title="调用工具: web_search",
            tool="web_search",
            tool_input={"query": "A股 最新 指数"},
        ),
        AgentEvent(
            type="tool_result",
            title="工具结果: web_search",
            tool="web_search",
            tool_output="上证指数 3000 点",
            content="上证指数 3000 点",
        ),
        AgentEvent(type="final", title="最终答案", content="上证指数 3000 点"),
    ]

    def fake_get_llm(tier: str, *, temperature: float):
        assert tier == "turbo"
        assert temperature == 0.1
        return object()

    def fake_run_react(prompt: str, *, llm: object, event_callback=None):
        assert "当前步骤：查询最新A股指数" in prompt
        for event in nested_events:
            if event_callback is not None:
                event_callback(event)
        return AgentRunResult(final_answer="上证指数 3000 点", events=nested_events, raw_state={})

    streamed_events = []
    monkeypatch.setattr(plan_execute, "get_llm", fake_get_llm)
    monkeypatch.setattr(plan_execute, "run_react", fake_run_react)

    result = run_plan_and_execute("最新的A股指数", planner=one_step_planner, event_callback=streamed_events.append)

    assert [event.type for event in streamed_events] == ["plan", "tool_call", "tool_result", "final", "step", "final"]
    assert streamed_events[1].title == "步骤 1 / ReAct 第 1 轮 / 调用工具: web_search"
    assert streamed_events[1].metadata["parent_step_id"] == 1
    assert streamed_events[2].title == "步骤 1 / ReAct 第 1 轮 / 工具结果: web_search"
    assert streamed_events[3].title == "步骤 1 / ReAct 第 2 轮 / 最终答案"
    assert streamed_events == result.events
    assert result.final_answer.endswith("上证指数 3000 点")
