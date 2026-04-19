"""DashScope 内置搜索 —— Web 兼底工具。

原理：调用 DashScope **原生** /api/v1/services/aigc/text-generation/generation 端点，
启用 `enable_search + enable_source`，response.output.search_info.search_results 返回 5-10 条
结构化搜索结果（title / url / site_name）。

为什么不用 OpenAI 兼容端点：兼容端点遵循 OpenAI schema，会丢弃 search_info 字段。

为什么放在 tools/：它符合 SearchTool 协议（snippet / source_url / relevance_score），对
Researcher 节点而言与 Tavily/Brave 完全等价，只是背后是 LLM + 内置 bing 搜索。

定位：Web 降级链的兜底（Tavily → DashScope Search → 跳过）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools._http import make_client, safe_post_json
from tools.base import SearchTool, ToolResult
from config.settings import settings

logger = logging.getLogger(__name__)
DASHSCOPE_GEN_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"


class DashScopeSearchTool:
    name = "dashscope-search"
    source_type = "web"

    def __init__(self, *, model: str = "qwen-turbo") -> None:
        self._model = model
        self._client = make_client(timeout=60)
        self._api_key = settings.dashscope_api_key

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        if not self._api_key:
            logger.info("[dashscope-search] api_key 未配置，跳过")
            return []

        payload = {
            "model": self._model,
            "input": {"messages": [{"role": "user", "content": query}]},
            "parameters": {
                "enable_search": True,
                "search_options": {
                    "forced_search": True,
                    "enable_source": True,
                    "enable_citation": False,  # 不需要 LLM 合成内容带 [N]，我们只要原始搜索结果
                    "search_strategy": "standard",
                },
                "result_format": "message",
                # 尽量少生成内容，省 token（我们只消费 search_info）
                "max_tokens": 50,
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        data = await safe_post_json(self._client, DASHSCOPE_GEN_URL, headers=headers, json=payload)
        if not data or not isinstance(data, dict):
            return []

        search_info = ((data.get("output") or {}).get("search_info") or {})
        raw_results: list[dict[str, Any]] = search_info.get("search_results") or []
        if not raw_results:
            return []

        out: list[ToolResult] = []
        n = max(1, len(raw_results))
        for i, r in enumerate(raw_results[:top_k]):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            title = (r.get("title") or "").strip()
            site = (r.get("site_name") or "").strip()
            snippet = f"《{title}》 · {site}" if site else title
            out.append(
                ToolResult(
                    snippet=snippet[:1500],
                    source_url=url,
                    relevance_score=max(0.1, 1.0 - (i / n) * 0.6),  # 前几条分高
                    extra={"title": title, "site_name": site, "index": r.get("index")},
                )
            )
        return out

    async def close(self) -> None:
        await self._client.aclose()


_: SearchTool = DashScopeSearchTool()
