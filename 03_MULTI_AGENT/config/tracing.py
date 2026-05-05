"""LangSmith tag helpers. 详见 ENGINEERING.md §10.1。"""
from __future__ import annotations

import functools
from typing import Any, Callable

from langchain_core.runnables import RunnableLambda


def with_tags(name: str, **extra) -> Callable:
    """给节点函数保留轻量标记，兼容旧的 researcher 装饰器。"""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        wrapper.__tags__ = {"agent": name, **extra}  # type: ignore[attr-defined]
        return wrapper

    return deco


def tagged_node(name: str, fn: Callable[..., Any], **metadata: Any):
    """把 LangGraph 节点包装为带 LangSmith tag/metadata 的 Runnable。

    LangGraph 会把 RunnableConfig 继续传给节点内的 LLM / tool 子 run；
    这里统一补上 agent 维度，便于 LangSmith 按节点过滤。
    """
    node_metadata = {"agent": name, **metadata}
    return RunnableLambda(fn, name=name).with_config(
        tags=[f"agent:{name}"],
        metadata=node_metadata,
    )
