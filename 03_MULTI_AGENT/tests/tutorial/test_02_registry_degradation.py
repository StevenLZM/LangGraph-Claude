"""配套 TUTORIAL.md 第 3 章 —— ToolRegistry 降级链

学什么：ToolRegistry.get_chain(source_type) 按注册顺序返回工具列表。
        run_research_chain 顺序调用每个工具，第一个抛异常或返空就降级到下一个。

断言：
1. 第一个工具异常 → 第二个工具被调用
2. 第二个工具返空 → 第三个工具兜底
3. 全部失败 → 返回空 list
4. 工具调用顺序严格按注册顺序
"""
from __future__ import annotations

import pytest

from agents._researcher_base import run_research_chain
from tools.registry import ToolRegistry


class _FakeTool:
    def __init__(self, name: str, *, results=None, raises: Exception | None = None):
        self.name = name
        self.source_type = "web"
        self._results = results or []
        self._raises = raises
        self.call_count = 0

    async def search(self, query: str, *, top_k: int = 5):
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        return self._results

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_first_raises_falls_back_to_second():
    t1 = _FakeTool("primary", raises=RuntimeError("api down"))
    t2 = _FakeTool("backup", results=[
        {"snippet": "ok", "source_url": "https://b/x", "relevance_score": 0.7}
    ])
    reg = ToolRegistry()
    reg.register(t1)
    reg.register(t2)

    evs = await run_research_chain(
        source_type="web", query="q", sub_question_id="sq1", registry=reg
    )
    assert t1.call_count == 1
    assert t2.call_count == 1
    assert len(evs) == 1 and evs[0].source_url == "https://b/x"


@pytest.mark.asyncio
async def test_empty_results_fall_through_to_third():
    t1 = _FakeTool("primary", results=[])
    t2 = _FakeTool("middle", results=[])
    t3 = _FakeTool("fallback", results=[
        {"snippet": "found", "source_url": "https://z/1", "relevance_score": 0.5}
    ])
    reg = ToolRegistry()
    for t in (t1, t2, t3):
        reg.register(t)

    evs = await run_research_chain(
        source_type="web", query="q", sub_question_id="sq1", registry=reg
    )
    assert (t1.call_count, t2.call_count, t3.call_count) == (1, 1, 1)
    assert len(evs) == 1 and evs[0].source_url == "https://z/1"


@pytest.mark.asyncio
async def test_all_fail_returns_empty():
    reg = ToolRegistry()
    reg.register(_FakeTool("a", raises=Exception("x")))
    reg.register(_FakeTool("b", results=[]))
    evs = await run_research_chain(
        source_type="web", query="q", sub_question_id="sq1", registry=reg
    )
    assert evs == []


@pytest.mark.asyncio
async def test_first_success_short_circuits():
    """第一个工具命中后，后续工具不再被调用（短路）。"""
    t1 = _FakeTool("primary", results=[
        {"snippet": "hit", "source_url": "https://p/1", "relevance_score": 0.9}
    ])
    t2 = _FakeTool("backup", results=[])
    reg = ToolRegistry()
    reg.register(t1)
    reg.register(t2)

    await run_research_chain(
        source_type="web", query="q", sub_question_id="sq1", registry=reg
    )
    assert t1.call_count == 1
    assert t2.call_count == 0, "首个工具命中后不应再调用 backup"
