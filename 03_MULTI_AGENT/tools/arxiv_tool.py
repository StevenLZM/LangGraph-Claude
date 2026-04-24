"""ArXiv 学术检索 —— 真实实现。

ArXiv API 返回 Atom XML：http://export.arxiv.org/api/query?search_query=...
无需 API Key，但有 ~3s 调用间隔约束。
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

from tools._http import make_client, safe_get_text
from tools.base import SearchTool, ToolResult

logger = logging.getLogger(__name__)
ARXIV_URL = "http://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ArxivTool:
    name = "arxiv"
    source_type = "academic"

    def __init__(self) -> None:
        self._client = make_client(timeout=30)

    async def search(self, query: str, *, top_k: int = 5) -> list[ToolResult]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": top_k,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        text = await safe_get_text(self._client, ARXIV_URL, params=params)
        if not text:
            return []
        try:
            root = ET.fromstring(text)
            print(f"arxiv原始结果:{str(root)[:100]}")
        except ET.ParseError as e:
            logger.warning("[arxiv] XML parse failed: %s", e)
            return []
        results: list[ToolResult] = []
        for i, entry in enumerate(root.findall("a:entry", NS)):
            title = (entry.findtext("a:title", default="", namespaces=NS) or "").strip()
            summary = (entry.findtext("a:summary", default="", namespaces=NS) or "").strip()
            link_el = entry.find("a:id", NS)
            url = (link_el.text or "").strip() if link_el is not None else ""
            authors = [
                (a.findtext("a:name", default="", namespaces=NS) or "").strip()
                for a in entry.findall("a:author", NS)
            ]
            results.append(
                ToolResult(
                    snippet=f"《{title}》\n作者: {', '.join(authors[:5])}\n摘要: {summary}"[:1800],
                    source_url=url,
                    relevance_score=max(0.0, 1.0 - i * 0.1),  # 按返回顺序衰减
                    extra={"title": title, "authors": authors},
                )
            )
        print(f"arxiv处理后结果:{str(results)[:100]}")
        return results

    async def close(self) -> None:
        await self._client.aclose()
