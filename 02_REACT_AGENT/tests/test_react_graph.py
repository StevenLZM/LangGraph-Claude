from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agent.react import build_react_graph, run_react
from tools.builtin import calculator, get_datetime


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


class _RecencyAwareLLM:
    def __init__(self):
        self.calls = 0
        self.search_query = ""
        self.first_invoke_had_datetime = False

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            self.first_invoke_had_datetime = any(
                isinstance(message, ToolMessage) and message.name == "get_datetime"
                for message in messages
            )
            system_prompt = str(messages[0].content)
            assert "真实当前日期时间" in system_prompt
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_datetime",
                        "args": {"timezone": "Asia/Shanghai"},
                        "id": "datetime-1",
                    }
                ],
            )

        datetime_results = [
            message
            for message in messages
            if isinstance(message, ToolMessage) and message.name == "get_datetime"
        ]
        if not datetime_results:
            return AIMessage(content="缺少当前时间，无法安全搜索最新指数。")
        if self.calls == 2:
            date_text = str(datetime_results[-1].content)
            match = re.search(r"(\d{4})年(\d{2})月(\d{2})日", date_text)
            assert match, date_text
            self.search_query = f"A股 最新 指数 {match.group(1)}-{match.group(2)}-{match.group(3)}"
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": self.search_query, "max_results": 3},
                        "id": "search-1",
                    }
                ],
            )
        return AIMessage(content=f"已基于搜索查询回答：{self.search_query}")


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


def test_run_react_streams_events_through_callback_in_execution_order():
    streamed_events = []

    result = run_react(
        "1234 * 5678 等于多少？",
        llm=_ToolCallingLLM(),
        tools=[calculator],
        event_callback=streamed_events.append,
    )

    assert [event.type for event in streamed_events] == ["tool_call", "tool_result", "final"]
    assert streamed_events == result.events
    assert result.final_answer == "1234 * 5678 = 7006652"


def test_run_react_lets_llm_get_current_datetime_before_searching_latest_query():
    @tool("web_search")
    def fake_web_search(query: str, max_results: int = 3) -> str:
        """Fake search tool."""
        return f"搜索查询: {query}; max_results={max_results}"

    llm = _RecencyAwareLLM()

    result = run_react(
        "A股最新指数是多少",
        llm=llm,
        tools=[get_datetime, fake_web_search],
        max_iterations=5,
    )

    calls = [event for event in result.events if event.type == "tool_call"]
    assert [event.tool for event in calls] == ["get_datetime", "web_search"]
    assert llm.first_invoke_had_datetime is False
    assert "A股 最新 指数" in llm.search_query
    assert re.search(r"\d{4}-\d{2}-\d{2}", llm.search_query)
