"""InsightLoop Streamlit 单页 UI（M5）。

设计：plans/tidy-hatching-gem.md §3
启动后端：
    PYTHONPATH=. uvicorn app.api:app --port 8080
启动 UI：
    streamlit run app/streamlit_ui.py
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from httpx_sse import connect_sse

API_BASE = os.environ.get("INSIGHTLOOP_API", "http://localhost:8080")

# graph 顺序（plans/tidy-hatching-gem.md）：planner → supervisor → researchers* → reflector → writer
NODE_ORDER = (
    "planner",
    "supervisor",
    "web_researcher",
    "academic_researcher",
    "code_researcher",
    "kb_researcher",
    "reflector",
    "writer",
)
NODE_LABELS = {
    "planner": "🧠 Planner",
    "supervisor": "🎯 Supervisor",
    "web_researcher": "🌐 Web Researcher",
    "academic_researcher": "📚 Academic Researcher",
    "code_researcher": "💻 Code Researcher",
    "kb_researcher": "📂 KB Researcher",
    "reflector": "🔍 Reflector",
    "writer": "✍️ Writer",
}
SOURCE_TO_NODE = {
    "web": "web_researcher",
    "academic": "academic_researcher",
    "code": "code_researcher",
    "kb": "kb_researcher",
}
ALWAYS_VISIBLE = {"planner", "supervisor", "reflector", "writer"}


def _init_state():
    s = st.session_state
    s.setdefault("thread_id", None)
    s.setdefault("history", [])  # list[{"role","content"}]
    s.setdefault("events", [])  # list[{"node","status","detail"}]
    s.setdefault("node_status", {})  # node -> "running"|"done"
    s.setdefault("report_buffer", "")
    s.setdefault("report_path", None)
    s.setdefault("pending_interrupt", None)
    s.setdefault("plan_df", None)
    s.setdefault("active_nodes", None)  # None = 显示全部默认；否则按 graph 顺序过滤
    s.setdefault("audience", "intermediate")
    s.setdefault("submitted_query", None)


def _active_nodes_from_plan(plan: dict | None) -> list[str]:
    """从 plan 抽 researcher 节点 + 固定节点，返回 graph 顺序排列的 node 列表。"""
    used: set[str] = set(ALWAYS_VISIBLE)
    if plan:
        for sq in plan.get("sub_questions", []) or []:
            for src in sq.get("recommended_sources", []) or []:
                node = SOURCE_TO_NODE.get(src)
                if node:
                    used.add(node)
    return [n for n in NODE_ORDER if n in used]


def _reset_run():
    st.session_state.events = []
    st.session_state.node_status = {}
    st.session_state.report_buffer = ""
    st.session_state.report_path = None
    st.session_state.pending_interrupt = None
    st.session_state.plan_df = None
    st.session_state.active_nodes = None


def _push_event(node: str, status: str, detail: str = ""):
    st.session_state.events.append({"node": node, "status": status, "detail": detail})
    st.session_state.node_status[node] = status


def _render_sidebar(placeholder):
    s = st.session_state
    nodes = s.active_nodes or list(NODE_ORDER)  # plan 未确定前显示完整 graph
    lines = []
    for node in nodes:
        label = NODE_LABELS.get(node, node)
        status = s.node_status.get(node)
        if status == "done":
            icon = "✅"
        elif status == "waiting":
            icon = "⏸️"
        elif status == "running":
            icon = "⏳"
        else:
            icon = "⚪️"
        lines.append(f"{icon} {label}")
    tail = [ev for ev in s.events if ev["status"] == "tool"][-5:]
    block = "\n".join(lines)
    if tail:
        block += "\n\n**最近工具调用**\n" + "\n".join(
            f"- `{ev['node']}` → {ev['detail']}" for ev in tail
        )
    placeholder.markdown(block)


def _consume_stream(url: str, params: dict, report_ph, side_ph) -> dict:
    """同步消费 SSE 流，更新两个 placeholder；返回 {final_report, report_path, interrupt}。"""
    final: dict[str, Any] = {}
    flush_every = 12
    token_counter = 0
    with httpx.Client(timeout=None) as client:
        with connect_sse(client, "GET", url, params=params) as event_source:
            for sse_ev in event_source.iter_sse():
                etype = sse_ev.event
                try:
                    data = json.loads(sse_ev.data) if sse_ev.data else {}
                except json.JSONDecodeError:
                    data = {"raw": sse_ev.data}

                if etype == "thread":
                    st.session_state.thread_id = data.get("thread_id")
                elif etype == "node_start":
                    _push_event(data["node"], "running")
                    _render_sidebar(side_ph)
                elif etype == "node_end":
                    summary = data.get("summary") or {}
                    detail = " ".join(f"{k}={v}" for k, v in summary.items())
                    _push_event(data["node"], "done", detail)
                    _render_sidebar(side_ph)
                elif etype == "tool":
                    _push_event(
                        data.get("node", "?"),
                        "tool",
                        f"{data.get('tool')} ({data.get('phase')})",
                    )
                    _render_sidebar(side_ph)
                elif etype == "token":
                    st.session_state.report_buffer += data.get("text", "")
                    token_counter += 1
                    if token_counter % flush_every == 0:
                        report_ph.markdown(st.session_state.report_buffer)
                elif etype == "interrupt":
                    final["interrupt"] = data
                    # interrupt 截断节点不发 on_chain_end，把当前 running 的节点标记为等待确认
                    for node, status in list(st.session_state.node_status.items()):
                        if status == "running":
                            st.session_state.node_status[node] = "waiting"
                    _render_sidebar(side_ph)
                    return final
                elif etype == "error":
                    st.error(f"后端错误: {data.get('message')}")
                    return final
                elif etype == "done":
                    final["final_report"] = data.get("final_report")
                    final["report_path"] = data.get("report_path")
                    if final.get("final_report"):
                        st.session_state.report_buffer = final["final_report"]
                    report_ph.markdown(st.session_state.report_buffer)
                    return final
    return final


def _plan_to_df(plan: dict) -> pd.DataFrame:
    rows = []
    for sq in plan.get("sub_questions", []):
        rows.append({
            "id": sq.get("id", ""),
            "question": sq.get("question", ""),
            "sources": ",".join(sq.get("recommended_sources", [])),
        })
    return pd.DataFrame(rows)


def _df_to_plan(df: pd.DataFrame, original: dict) -> dict:
    valid_sources = {"web", "academic", "code", "kb"}
    sub_questions = []
    for _, row in df.iterrows():
        srcs = [s.strip() for s in str(row["sources"]).split(",") if s.strip() in valid_sources]
        if not srcs:
            srcs = ["web"]
        sub_questions.append({
            "id": str(row["id"]) or f"sq_{len(sub_questions)+1}",
            "question": str(row["question"]).strip(),
            "recommended_sources": srcs,
            "status": "pending",
        })
    return {
        "sub_questions": sub_questions,
        "estimated_depth": original.get("estimated_depth", "standard"),
    }


def main():
    st.set_page_config(page_title="InsightLoop", layout="wide")
    _init_state()

    st.title("InsightLoop · 多 Agent 深度研究")
    top_l, top_r = st.columns([6, 2])
    with top_l:
        query = st.text_input("研究问题", key="query_input", placeholder="例：对比 LangGraph 与 AutoGen 的核心抽象差异")
    with top_r:
        st.session_state.audience = st.selectbox(
            "目标读者",
            ["beginner", "intermediate", "expert"],
            index=["beginner", "intermediate", "expert"].index(st.session_state.audience),
        )
    submit = st.button("🚀 开始研究", type="primary", disabled=not query.strip())

    body_l, body_r = st.columns([8, 4])
    with body_r:
        st.markdown("### 实时进度")
        side_ph = st.empty()
        _render_sidebar(side_ph)
    with body_l:
        st.markdown("### 报告")
        report_ph = st.empty()
        if st.session_state.report_buffer:
            report_ph.markdown(st.session_state.report_buffer)

    if submit:
        _reset_run()
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.submitted_query = query
        result = _consume_stream(
            f"{API_BASE}/research/stream",
            {"query": query, "audience": st.session_state.audience},
            report_ph,
            side_ph,
        )
        if intr := result.get("interrupt"):
            plan = (intr or {}).get("plan", {})
            st.session_state.pending_interrupt = plan
            st.session_state.plan_df = _plan_to_df(plan)
            st.session_state.active_nodes = _active_nodes_from_plan(plan)
            _render_sidebar(side_ph)
        elif result.get("report_path"):
            st.session_state.report_path = result["report_path"]

    # 计划编辑面板：用 placeholder 包住，按"确认"后立即清空避免按钮残留
    plan_panel = st.empty()
    if st.session_state.pending_interrupt:
        with plan_panel.container():
            with st.expander("✏️ 编辑研究计划", expanded=True):
                st.caption("可修改子问题描述或推荐源（逗号分隔，取值：web/academic/code/kb）")
                edited = st.data_editor(
                    st.session_state.plan_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="plan_editor",
                )
                confirm = st.button("✅ 确认计划，继续研究", key="confirm_plan_btn")
        if confirm:
            plan_payload = _df_to_plan(edited, st.session_state.pending_interrupt)
            st.session_state.pending_interrupt = None
            st.session_state.plan_df = None
            st.session_state.active_nodes = _active_nodes_from_plan(plan_payload)
            plan_panel.empty()  # 立即清掉 expander + 按钮，避免阻塞流期间还在屏上
            _render_sidebar(side_ph)
            result = _consume_stream(
                f"{API_BASE}/research/{st.session_state.thread_id}/resume_stream",
                {"plan": json.dumps(plan_payload, ensure_ascii=False)},
                report_ph,
                side_ph,
            )
            if result.get("report_path"):
                st.session_state.report_path = result["report_path"]

    if st.session_state.report_path:
        st.success(f"📄 报告已归档：`{st.session_state.report_path}`")

    if st.session_state.thread_id and not st.session_state.pending_interrupt:
        st.divider()
        followup = st.text_input(
            "💬 继续追问（同会话复用历史 evidence）",
            key="followup_input",
            placeholder="例：那它们在生产部署上的差异呢？",
        )
        if st.button("追问", disabled=not followup.strip()):
            _reset_run()
            st.session_state.history.append({"role": "user", "content": followup})
            result = _consume_stream(
                f"{API_BASE}/research/{st.session_state.thread_id}/turn_stream",
                {"query": followup},
                report_ph,
                side_ph,
            )
            if intr := result.get("interrupt"):
                plan = (intr or {}).get("plan", {})
                st.session_state.pending_interrupt = plan
                st.session_state.plan_df = _plan_to_df(plan)
                st.session_state.active_nodes = _active_nodes_from_plan(plan)
                _render_sidebar(side_ph)
            elif result.get("report_path"):
                st.session_state.report_path = result["report_path"]


if __name__ == "__main__":
    main()
