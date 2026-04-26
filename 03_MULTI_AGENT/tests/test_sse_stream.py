"""SSE 事件映射 + API 端点冒烟测试 —— M5。

不依赖真实 LLM / 真实 LangGraph 运行；用伪事件流验证：
  - app/sse.map_event 的字段映射规则（writer-only token、节点过滤、interrupt 提取）
  - app/api 三条 SSE 端点能正确把事件流转成 SSE 协议响应
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app import sse


def test_map_event_node_start_filters_internal_runs():
    assert sse.map_event({"event": "on_chain_start", "name": "planner", "data": {}}) == {
        "event": "node_start",
        "data": {"node": "planner"},
    }
    # 非业务节点（langgraph 内部 run）应忽略
    assert sse.map_event({"event": "on_chain_start", "name": "ChannelWrite", "data": {}}) is None


def test_map_event_node_end_summary():
    out = sse.map_event(
        {
            "event": "on_chain_end",
            "name": "reflector",
            "data": {
                "output": {
                    "next_action": "sufficient",
                    "revision_count": 1,
                    "evidence": [1, 2, 3],
                    "messages": ["巨大对象不应外泄"],
                }
            },
        }
    )
    assert out["event"] == "node_end"
    assert out["data"]["node"] == "reflector"
    summary = out["data"]["summary"]
    assert summary["next_action"] == "sufficient"
    assert summary["evidence_count"] == 3
    assert "messages" not in summary


def test_map_event_token_only_for_writer():
    class _Chunk:
        content = "hello"

    writer_ev = {
        "event": "on_chat_model_stream",
        "name": "ChatDeepSeek",
        "metadata": {"langgraph_node": "writer"},
        "data": {"chunk": _Chunk()},
    }
    assert sse.map_event(writer_ev) == {"event": "token", "data": {"text": "hello"}}

    planner_ev = dict(writer_ev)
    planner_ev["metadata"] = {"langgraph_node": "planner"}
    assert sse.map_event(planner_ev) is None


def test_map_event_interrupt_from_root_graph():
    class _Intr:
        value = {"phase": "plan_review", "plan": {"sub_questions": []}}

    ev = {
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"__interrupt__": [_Intr()]}},
    }
    assert sse.map_event(ev) == {
        "event": "interrupt",
        "data": {"phase": "plan_review", "plan": {"sub_questions": []}},
    }


def test_coerce_plan_payload_accepts_string_and_dict():
    plan = {
        "sub_questions": [
            {"id": "sq1", "question": "q", "recommended_sources": ["web"], "status": "pending"}
        ],
        "estimated_depth": "quick",
    }
    assert sse.coerce_plan_payload(plan)["plan"]["sub_questions"][0]["id"] == "sq1"
    assert sse.coerce_plan_payload(json.dumps(plan))["plan"]["estimated_depth"] == "quick"
    assert sse.coerce_plan_payload({"plan": plan})["plan"]["sub_questions"][0]["question"] == "q"


async def _fake_stream(events: list[dict]) -> AsyncIterator[dict]:
    for ev in events:
        yield ev


@pytest.mark.asyncio
async def test_stream_events_emits_thread_then_done():
    events = [
        {"event": "on_chain_start", "name": "planner", "data": {}},
        {"event": "on_chain_end", "name": "planner", "data": {"output": {"plan_confirmed": True}}},
        {"event": "on_chain_end", "name": "LangGraph",
         "data": {"output": {"final_report": "# r", "report_path": "/tmp/r.md"}}},
    ]
    out = []
    async for item in sse.stream_events(_fake_stream(events), thread_id="tid123"):
        out.append(item)
    assert out[0] == {"event": "thread", "data": {"thread_id": "tid123"}}
    assert any(o["event"] == "node_start" for o in out)
    assert any(o["event"] == "node_end" for o in out)
    assert out[-1]["event"] == "done"
    assert out[-1]["data"]["report_path"] == "/tmp/r.md"


@pytest.mark.asyncio
async def test_stream_events_skips_done_when_interrupted():
    """流里没 __interrupt__，但 aget_state 返回挂起的 interrupt → 仍要发 interrupt 事件。"""

    class _Intr:
        value = {"phase": "plan_review"}

    class _Task:
        interrupts = (_Intr(),)

    class _State:
        tasks = (_Task(),)

    class _FakeGraph:
        async def aget_state(self, cfg):
            return _State()

    events = [
        {"event": "on_chain_start", "name": "planner", "data": {}},
        {"event": "on_chain_end", "name": "LangGraph", "data": {"output": {"q": "hi"}}},
    ]
    out = []
    async for item in sse.stream_events(
        _fake_stream(events), thread_id="t", graph=_FakeGraph(), cfg={"configurable": {}}
    ):
        out.append(item)
    types = [o["event"] for o in out]
    assert "interrupt" in types
    assert "done" not in types
    intr = next(o for o in out if o["event"] == "interrupt")
    assert intr["data"] == {"phase": "plan_review"}


@pytest.mark.asyncio
async def test_research_stream_endpoint_smoke(monkeypatch):
    """通过 ASGITransport 调真实端点，但 graph 用 fake 替换。"""
    from app import api, bootstrap

    class _FakeGraph:
        def astream_events(self, payload, config, version):
            assert version == "v2"

            async def gen():
                yield {"event": "on_chain_start", "name": "planner", "data": {}}
                yield {"event": "on_chain_end", "name": "planner",
                       "data": {"output": {"plan_confirmed": True}}}
                yield {"event": "on_chain_end", "name": "LangGraph",
                       "data": {"output": {"final_report": "# r", "report_path": "/tmp/r.md"}}}

            return gen()

        async def aget_state(self, cfg):
            class _S:
                tasks = ()
            return _S()

    bootstrap.app_state.graph = _FakeGraph()

    transport = ASGITransport(app=api.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("GET", "/research/stream", params={"query": "q"}) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = b""
            async for chunk in resp.aiter_bytes():
                body += chunk
            text = body.decode()

    assert "event: thread" in text
    assert "event: node_start" in text
    assert "event: done" in text
    assert "/tmp/r.md" in text
