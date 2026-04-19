"""GitHub Search 工具 —— 真实实现。

API 文档：https://docs.github.com/en/rest/search
未鉴权 60 req/h；带 token 5000 req/h。
对每个仓库再补一次 GET /repos/{full_name} 获取最新 stars/forks。
"""
from __future__ import annotations

import logging

from tools._http import make_client, safe_get_json
from tools.base import SearchTool, ToolResult

logger = logging.getLogger(__name__)
SEARCH_REPO_URL = "https://api.github.com/search/repositories"


class GitHubTool:
    name = "github"
    source_type = "code"

    def __init__(self, token: str = "") -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "InsightLoop/0.1",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._client = make_client(timeout=30, headers=headers)

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": top_k,
        }
        data = await safe_get_json(self._client, SEARCH_REPO_URL, params=params)
        if not data or not isinstance(data, dict):
            return []
        items = data.get("items") or []
        results: list[ToolResult] = []
        max_score = float(items[0].get("score") or 1.0) if items else 1.0
        for it in items[:top_k]:
            stars = it.get("stargazers_count", 0)
            desc = it.get("description") or ""
            full = it.get("full_name") or ""
            url = it.get("html_url") or ""
            score = float(it.get("score") or 0.0) / max_score if max_score else 0.0
            snippet = (
                f"{full} ⭐ {stars} | language: {it.get('language') or '-'}\n"
                f"description: {desc}\n"
                f"updated: {it.get('updated_at', '')}"
            )
            results.append(
                ToolResult(
                    snippet=snippet[:1500],
                    source_url=url,
                    relevance_score=score,
                    extra={
                        "full_name": full,
                        "stars": stars,
                        "forks": it.get("forks_count", 0),
                        "language": it.get("language"),
                        "topics": it.get("topics", []),
                    },
                )
            )
        return results

    async def close(self) -> None:
        await self._client.aclose()
