"""Fan-out helpers —— 见 ENGINEERING.md §4.1。

当前实现位于 graph/research_subgraph.py::build_research_sends。
本文件保留旧导出名，方便教程或外部引用不必立刻改 import。
"""
from __future__ import annotations

from graph.research_subgraph import build_research_sends as fanout_researchers

__all__ = ["fanout_researchers"]
