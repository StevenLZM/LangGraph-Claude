from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.events import AgentEvent, AgentRunResult
from agent.prompts import REACT_SYSTEM_PROMPT
from config.llm import get_llm
from config.settings import settings
from tools.builtin import get_tools


class ReactState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int


def build_react_graph(llm: Any | None = None, tools: list[Any] | None = None, max_iterations: int | None = None):
    tool_list = tools or get_tools()
    bound_llm = llm or get_llm("max", temperature=0.1)
    if hasattr(bound_llm, "bind_tools"):
        bound_llm = bound_llm.bind_tools(tool_list)
    max_iters = max_iterations or settings.max_react_iterations

    def agent_node(state: ReactState) -> ReactState:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=REACT_SYSTEM_PROMPT), *messages]
        response = bound_llm.invoke(messages)
        return {
            "messages": [response],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }

    def should_continue(state: ReactState) -> str:
        if int(state.get("iteration_count", 0)) >= max_iters:
            return "end"
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        return "tools" if tool_calls else "end"

    workflow = StateGraph(ReactState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tool_list))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def _events_from_messages(messages: list[BaseMessage]) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    for message in messages:
        if isinstance(message, SystemMessage | HumanMessage):
            continue
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    events.append(
                        AgentEvent(
                            type="tool_call",
                            title=f"调用工具: {call.get('name')}",
                            tool=call.get("name"),
                            tool_input=call.get("args"),
                            metadata={"id": call.get("id")},
                        )
                    )
            elif message.content:
                events.append(AgentEvent(type="final", title="最终答案", content=str(message.content)))
        elif isinstance(message, ToolMessage):
            events.append(
                AgentEvent(
                    type="tool_result",
                    title=f"工具结果: {message.name or 'tool'}",
                    tool=message.name,
                    tool_output=str(message.content),
                    content=str(message.content),
                    metadata={"tool_call_id": message.tool_call_id},
                )
            )
    return events


def run_react(
    user_input: str,
    *,
    llm: Any | None = None,
    tools: list[Any] | None = None,
    max_iterations: int | None = None,
) -> AgentRunResult:
    graph = build_react_graph(llm=llm, tools=tools, max_iterations=max_iterations)
    state = graph.invoke({"messages": [HumanMessage(content=user_input)], "iteration_count": 0})
    events = _events_from_messages(state["messages"])
    final_answer = ""
    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage) and not (getattr(message, "tool_calls", None) or []):
            final_answer = str(message.content)
            break
    if not final_answer:
        final_answer = "达到最大工具调用步数，未能生成最终答案。"
        events.append(AgentEvent(type="error", title="执行停止", content=final_answer))
    return AgentRunResult(final_answer=final_answer, events=events, raw_state=dict(state))
