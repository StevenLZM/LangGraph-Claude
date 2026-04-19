"""配套 TUTORIAL.md 第 8 章 —— safe_node 容错装饰器

学什么：safe_node 把 Researcher / Reflector 等节点的异常转换为
        "空 evidence + 警告消息" 的合规返回值，避免单点失败中断整个图。

断言：
1. 被装饰函数抛异常 → wrapper 返回 dict，含 evidence=[] 与 messages
2. messages 中的内容包含函数名 + 异常文本（便于线上排查）
3. 正常路径透传返回值
4. 装饰器保留原函数名（functools.wraps）
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agents._safe import safe_node


@safe_node
async def _failing_node(state):
    raise ValueError("boom")


@safe_node
async def _ok_node(state):
    return {"evidence": [{"source_url": "https://x"}], "messages": []}


@pytest.mark.asyncio
async def test_exception_becomes_empty_evidence():
    out = await _failing_node({})
    assert out["evidence"] == []
    assert isinstance(out["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_message_carries_function_name_and_error():
    out = await _failing_node({})
    msg = out["messages"][0].content
    assert "_failing_node" in msg
    assert "boom" in msg


@pytest.mark.asyncio
async def test_happy_path_passthrough():
    out = await _ok_node({})
    assert out["evidence"] == [{"source_url": "https://x"}]


def test_preserves_function_name():
    assert _failing_node.__name__ == "_failing_node"
