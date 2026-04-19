"""Fan-out helpers —— 见 ENGINEERING.md §4.1。

当前实现合并在 graph/router.py::supervisor_route 中（Send 列表直接作为 conditional edge 返回）。
本文件保留作为后续可能抽离的接口占位。
"""
from __future__ import annotations

from graph.router import supervisor_route as fanout_researchers  # re-export

__all__ = ["fanout_researchers"]
