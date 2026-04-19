"""Brave Search MCP 适配器 —— 官方 mcp SDK 实现。

生命周期：
  - 连接由 AsyncExitStack 管理，进程存活期间保持一个 ClientSession
  - FastAPI shutdown 时通过 close() 触发 exit stack 关闭子进程

为什么保留单独 ClientSession 而非每次 search 重建：
  - stdio 子进程冷启动（npx 下载 + node 启动）耗时 1-3 秒；高频搜索时分摊不划算
  - mcp SDK 的 ClientSession 自带请求锁，并发 call_tool 是安全的
"""
from __future__ import annotations

import logging
import os
import re
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tools.base import ToolResult

logger = logging.getLogger(__name__)


class MCPBraveSearchTool:
    name = "mcp-brave"
    source_type = "web"

    def __init__(
        self,
        api_key: str,
        *,
        command: list[str] | None = None,
        tool_name: str = "brave_web_search",
        proxy: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("BRAVE_API_KEY 不能为空")
        self._api_key = api_key
        self._tool_name = tool_name
        env = {**os.environ, "BRAVE_API_KEY": api_key}
        # MCP 子进程是 Node 跑 fetch()，需要走代理才能到 api.search.brave.com（国内）
        if proxy:
            env["HTTPS_PROXY"] = proxy
            env["HTTP_PROXY"] = proxy
            # node 18+ undici 走 fetch 时还认 GLOBAL_AGENT/UNDICI_PROXY，但 HTTPS_PROXY 一般够用
        cmd = command or ["npx", "-y", "@modelcontextprotocol/server-brave-search"]
        self._params = StdioServerParameters(
            command=cmd[0],
            args=cmd[1:],
            env=env,
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(self._params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._stack = stack
            self._session = session
            logger.info("[mcp-brave] 会话已建立")
            return session
        except Exception:
            await stack.aclose()
            raise

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        try:
            session = await self._ensure_session()
            result = await session.call_tool(self._tool_name, {"query": query, "count": top_k})
        except Exception as e:
            logger.warning("[mcp-brave] 调用失败: %s", e)
            return []

        text = _extract_text(result)
        return _parse_brave_text(text, top_k)

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as e:
                logger.warning("[mcp-brave] close 异常: %s", e)
        self._stack = None
        self._session = None


# ────────────────────────────────────────────────────────────────────
# Brave MCP 输出解析（保留同一套解析逻辑）


def _extract_text(result: Any) -> str:
    """从 mcp.types.CallToolResult 或 dict 里合并所有 text content。"""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    parts: list[str] = []
    for c in content or []:
        text = getattr(c, "text", None)
        if text is None and isinstance(c, dict) and c.get("type") == "text":
            text = c.get("text")
        if text:
            parts.append(text)
    return "\n\n".join(parts)


_TITLE = re.compile(r"^Title:\s*(.+)$", re.IGNORECASE)
_DESC = re.compile(r"^Description:\s*(.+)$", re.IGNORECASE)
_URL = re.compile(r"^URL:\s*(\S+)$", re.IGNORECASE)


def _parse_brave_text(text: str, top_k: int) -> list[ToolResult]:
    if not text:
        return []
    results: list[ToolResult] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    n = max(1, min(len(blocks), top_k))
    for i, block in enumerate(blocks[:top_k]):
        title = ""
        desc = ""
        url = ""
        for line in block.splitlines():
            line = line.strip()
            if m := _TITLE.match(line):
                title = m.group(1).strip()
            elif m := _DESC.match(line):
                desc = m.group(1).strip()
            elif m := _URL.match(line):
                url = m.group(1).strip()
        if not url:
            continue
        snippet = f"《{title}》\n{desc}" if desc else title
        results.append(
            ToolResult(
                snippet=snippet[:1500],
                source_url=url,
                relevance_score=max(0.1, 1.0 - i * (0.6 / n)),
                extra={"title": title},
            )
        )
    return results
