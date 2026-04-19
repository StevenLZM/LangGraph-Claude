"""LangSmith tag 装饰器 —— 骨架。详见 ENGINEERING.md §10.1。"""
from __future__ import annotations

import functools
from typing import Callable


def with_tags(name: str, **extra) -> Callable:
    """给节点函数注入 LangSmith tag。M1 仅打印结构化日志。"""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            # TODO M6: 接 LangSmith tag 注入
            return await fn(*args, **kwargs)

        wrapper.__tags__ = {"agent": name, **extra}  # type: ignore[attr-defined]
        return wrapper

    return deco
