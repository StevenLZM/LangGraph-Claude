from __future__ import annotations

import json
from typing import Any

from agent.events import AgentEvent
from agent.plan_execute import run_plan_and_execute
from agent.react import run_react


def format_event_for_display(event: AgentEvent) -> dict[str, str]:
    if event.type == "tool_call":
        return {
            "label": event.title,
            "body": json.dumps(event.tool_input, ensure_ascii=False, indent=2),
            "language": "json",
        }
    if event.type == "tool_result":
        return {"label": event.title, "body": event.tool_output or event.content, "language": "text"}
    if event.type == "plan":
        return {"label": event.title, "body": event.content, "language": "markdown"}
    if event.type == "step":
        return {"label": event.title, "body": event.content, "language": "markdown"}
    if event.type == "error":
        return {"label": event.title, "body": event.content, "language": "text"}
    return {"label": event.title or "最终答案", "body": event.content, "language": "markdown"}


def _render_event(st: Any, event: AgentEvent) -> None:
    display = format_event_for_display(event)
    expanded = event.type in {"tool_call", "tool_result", "plan", "step", "error"}
    with st.expander(display["label"], expanded=expanded):
        if display["language"] == "json":
            st.code(display["body"], language="json")
        elif display["language"] == "markdown":
            st.markdown(display["body"])
        else:
            st.text(display["body"])


def _run(mode: str, prompt: str):
    if mode == "ReAct":
        return run_react(prompt)
    return run_plan_and_execute(prompt)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReAct Agent 工具助手", page_icon="🧭", layout="wide")
    st.title("ReAct Agent 工具助手")

    with st.sidebar:
        mode = st.radio("执行模式", ["ReAct", "Plan-and-Execute"], horizontal=False)
        st.caption("天气工具使用内部 MCP 实现；LLM 使用 DeepSeek OpenAI-compatible API。")
        if "last_events" in st.session_state:
            st.divider()
            st.subheader("推理链")
            for event in st.session_state.last_events:
                _render_event(st, event)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("输入任务，例如：今天北京天气怎么样，适合跑步吗？")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("执行中", expanded=True):
            try:
                result = _run(mode, prompt)
            except Exception as exc:
                result = None
                st.error(f"执行失败: {exc}")
            if result is not None:
                st.session_state.last_events = result.events
                for event in result.events:
                    _render_event(st, event)
                st.markdown(result.final_answer)
                st.session_state.messages.append({"role": "assistant", "content": result.final_answer})


if __name__ == "__main__":
    main()
