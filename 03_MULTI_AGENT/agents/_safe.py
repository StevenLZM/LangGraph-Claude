"""safe_node 装饰器 —— 节点级错误兜底。详见 ENGINEERING.md §4.3。

单 sub-agent 失败 → 返回空 evidence + 警告消息，不中断主图。
"""
from __future__ import annotations

import functools
import logging

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def safe_node(fn):
    @functools.wraps(fn)
    async def wrapper(state, *args, **kwargs):
        try:
            return await fn(state, *args, **kwargs)
        except Exception as e:  # pragma: no cover (覆盖见 M3 单测)
            logger.warning("node %s failed: %s", fn.__name__, e, exc_info=True)
            return {
                "evidence": [],
                "messages": [AIMessage(content=f"[skip] {fn.__name__}: {e}")],
            }

    return wrapper
