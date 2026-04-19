"""Registry / SearchTool 协议契约测试。"""
from __future__ import annotations

import pytest

from tools.base import SearchTool
from tools.registry import ToolRegistry


class _Fake:
    def __init__(self, name: str, st: str, results):
        self.name, self.source_type, self._r = name, st, results

    async def search(self, query, *, top_k=5):
        return self._r

    async def close(self):
        pass


def test_registry_get_chain_returns_in_order():
    reg = ToolRegistry()
    a, b = _Fake("a", "web", []), _Fake("b", "web", [{"source_url": "u"}])
    reg.register(a)
    reg.register(b)
    chain = reg.get_chain("web")
    assert [t.name for t in chain] == ["a", "b"]


def test_registry_unknown_source_returns_empty():
    reg = ToolRegistry()
    assert reg.get_chain("academic") == []


def test_fake_satisfies_protocol():
    f = _Fake("x", "web", [])
    assert isinstance(f, SearchTool)
