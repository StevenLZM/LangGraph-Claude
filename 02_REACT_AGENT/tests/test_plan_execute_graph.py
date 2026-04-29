from __future__ import annotations

from agent.plan_execute import Plan, PlanStep, StepResult, build_plan_execute_graph, run_plan_and_execute


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
