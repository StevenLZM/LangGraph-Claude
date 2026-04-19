"""Brave MCP smoke 脚本 —— 跳过 registry 链，直接对 brave-search MCP server 发一次请求。

用途：单独验证 Brave API key + npx + MCP 协议联通。

用法：
    cd 03_MULTI_AGENT
    PYTHONPATH=. python -m scripts.test_brave_mcp "LangGraph 2025"
"""
from __future__ import annotations

import asyncio
import logging
import sys

from config.settings import settings
from tools.mcp_brave_tool import MCPBraveSearchTool


async def main(query: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not settings.brave_api_key:
        print("❌ BRAVE_API_KEY 未配置")
        return

    tool = MCPBraveSearchTool(api_key=settings.brave_api_key)
    try:
        print(f"📡 调用 brave-search MCP，query={query!r} ...")
        results = await tool.search(query, top_k=5)
        print(f"\n✅ 命中 {len(results)} 条:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] score={r['relevance_score']:.2f}  url={r['source_url']}")
            print(f"    {r['snippet'][:200]}")
    finally:
        await tool.close()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "LangGraph 1.0 production"
    asyncio.run(main(q))
