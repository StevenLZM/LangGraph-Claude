"""Brave MCP 解析器测试（无需启动真实 server）。"""
from __future__ import annotations

from tools.mcp_brave_tool import _parse_brave_text


def test_brave_text_parser():
    text = """Title: First Result
Description: A description
URL: https://example.com/1

Title: Second
Description: Another
URL: https://example.com/2"""
    results = _parse_brave_text(text, top_k=5)
    assert len(results) == 2
    assert results[0]["source_url"] == "https://example.com/1"
    assert "First Result" in results[0]["snippet"]
    assert results[0]["relevance_score"] > results[1]["relevance_score"]


def test_brave_text_parser_skips_blocks_without_url():
    text = """Title: Only title
Description: no url here

Title: Has URL
URL: https://x"""
    results = _parse_brave_text(text, top_k=5)
    assert len(results) == 1
    assert results[0]["source_url"] == "https://x"


def test_brave_text_parser_empty():
    assert _parse_brave_text("", top_k=5) == []
