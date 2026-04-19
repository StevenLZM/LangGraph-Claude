"""Tavily Web Search 工具 —— 真实实现。

API 文档：https://docs.tavily.com/docs/rest-api/api-reference
免费档每分钟 10 次；建议拿 search_depth=basic。
"""
from __future__ import annotations

import logging

from tools._http import make_client, safe_post_json
from tools.base import SearchTool, ToolResult

logger = logging.getLogger(__name__)
TAVILY_URL = "https://api.tavily.com/search"


class TavilyTool:
    name = "tavily"
    source_type = "web"

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._client = make_client(timeout=30)

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        if not self._api_key:
            logger.info("[tavily] api_key 未配置，返回空")
            return []
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": False,
        }
        data = await safe_post_json(self._client, TAVILY_URL, json=payload)
        if not data or not isinstance(data, dict):
            return []
        results = []
        for r in data.get("results", [])[:top_k]:
            results.append(
                ToolResult(
                    snippet=(r.get("content") or "")[:1500],
                    source_url=r.get("url") or "",
                    relevance_score=float(r.get("score") or 0.0),
                    extra={"title": r.get("title", "")},
                )
            )
        return results

    async def close(self) -> None:
        await self._client.aclose()


_: SearchTool = TavilyTool()
