from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.react import build_react_graph, run_react
from tools.builtin import calculator


class _ToolCallingLLM:
    def __init__(self):
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": "1234 * 5678"},
                        "id": "calc-1",
                    }
                ],
            )
        return AIMessage(content="1234 * 5678 = 7006652")


def test_react_graph_calls_tool_then_returns_final_answer():
    graph = build_react_graph(llm=_ToolCallingLLM(), tools=[calculator], max_iterations=5)

    state = graph.invoke({"messages": [HumanMessage(content="1234 * 5678 等于多少？")], "iteration_count": 0})

    assert any(isinstance(message, ToolMessage) for message in state["messages"])
    assert state["messages"][-1].content == "1234 * 5678 = 7006652"
    assert state["iteration_count"] == 2


def test_run_react_returns_events_for_tool_call_and_final_answer():
    result = run_react("1234 * 5678 等于多少？", llm=_ToolCallingLLM(), tools=[calculator])

    event_types = [event.type for event in result.events]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert event_types[-1] == "final"
    assert result.final_answer == "1234 * 5678 = 7006652"
