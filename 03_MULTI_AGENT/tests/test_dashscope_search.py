"""Web 降级链测试：验证 registry 中两个 web 工具按顺序注册。"""
from __future__ import annotations

import pytest

from tools.dashscope_search_tool import DashScopeSearchTool
from tools.registry import ToolRegistry
from tools.tavily_tool import TavilyTool


def test_web_chain_order():
    reg = ToolRegistry()
    reg.register(TavilyTool(api_key="x"))
    reg.register(DashScopeSearchTool())
    chain = reg.get_chain("web")
    names = [t.name for t in chain]
    assert names == ["tavily", "dashscope-search"], f"web 降级链顺序错误: {names}"


@pytest.mark.asyncio
async def test_dashscope_search_no_key_returns_empty(monkeypatch):
    t = DashScopeSearchTool()
    # 临时清空 key
    t._api_key = ""
    result = await t.search("test")
    assert result == []
    await t.close()
