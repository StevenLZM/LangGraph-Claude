from __future__ import annotations

from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.events import AgentEvent, AgentRunResult
from agent.prompts import build_react_system_prompt
from config.llm import get_llm
from config.settings import settings
from tools.registry import get_tools


class ReactState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int


EventCallback = Callable[[AgentEvent], None]


def build_react_graph(llm: Any | None = None, tools: list[Any] | None = None, max_iterations: int | None = None):
    tool_list = tools or get_tools()
    system_prompt = build_react_system_prompt(tool_list)
    bound_llm = llm or get_llm("max", temperature=0.1)
    if hasattr(bound_llm, "bind_tools"):
        bound_llm = bound_llm.bind_tools(tool_list)
    max_iters = max_iterations or settings.max_react_iterations

    def agent_node(state: ReactState) -> ReactState:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
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
        print(f"llm message类型:{message.__class__}")
        print(f"llm message:{message}")
        if isinstance(message, SystemMessage | HumanMessage):
            continue
        # -graph里如果是LLM输出
        #   -有toolcall：调用工具(一次一个或多个)
        #   -没有toolcall：final answer
        # -graph里tool输出
        #   -工具结果(一个结果一条message)
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


def _final_answer_from_messages(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not (getattr(message, "tool_calls", None) or []):
            return str(message.content)
    return ""


def _emit_event(events: list[AgentEvent], event: AgentEvent, event_callback: EventCallback | None) -> None:
    events.append(event)
    if event_callback is not None:
        event_callback(event)


def _run_react_with_event_callback(
    user_input: str,
    *,
    llm: Any | None = None,
    tools: list[Any] | None = None,
    max_iterations: int | None = None,
    event_callback: EventCallback,
) -> AgentRunResult:
    graph = build_react_graph(llm=llm, tools=tools, max_iterations=max_iterations)
    messages: list[BaseMessage] = [HumanMessage(content=user_input)]
    state: ReactState = {"messages": messages, "iteration_count": 0}
    events: list[AgentEvent] = []

    for update in graph.stream(state, stream_mode="updates"):
        for node_state in update.values():
            new_messages = list(node_state.get("messages", []))
            if new_messages:
                messages.extend(new_messages)
                for event in _events_from_messages(new_messages):
                    _emit_event(events, event, event_callback)
            if "iteration_count" in node_state:
                state["iteration_count"] = int(node_state["iteration_count"])

    state["messages"] = messages
    final_answer = _final_answer_from_messages(messages)
    if not final_answer:
        final_answer = "达到最大工具调用步数，未能生成最终答案。"
        _emit_event(events, AgentEvent(type="error", title="执行停止", content=final_answer), event_callback)
    return AgentRunResult(final_answer=final_answer, events=events, raw_state=dict(state))


def run_react(
    user_input: str,
    *,
    llm: Any | None = None,
    tools: list[Any] | None = None,
    max_iterations: int | None = None,
    event_callback: EventCallback | None = None,
) -> AgentRunResult:
    if event_callback is not None:
        return _run_react_with_event_callback(
            user_input,
            llm=llm,
            tools=tools,
            max_iterations=max_iterations,
            event_callback=event_callback,
        )

    graph = build_react_graph(llm=llm, tools=tools, max_iterations=max_iterations)
    state = graph.invoke({"messages": [HumanMessage(content=user_input)], "iteration_count": 0})
    events = _events_from_messages(state["messages"])
    final_answer = _final_answer_from_messages(state["messages"])
    if not final_answer:
        final_answer = "达到最大工具调用步数，未能生成最终答案。"
        events.append(AgentEvent(type="error", title="执行停止", content=final_answer))
    return AgentRunResult(final_answer=final_answer, events=events, raw_state=dict(state))
