from __future__ import annotations

import sys
import types

from config.settings import settings
from tools.builtin import calculator, get_builtin_tools, get_datetime, python_executor, web_search, wikipedia_search


def test_calculator_evaluates_numeric_expression():
    result = calculator.invoke({"expression": "1234 * 5678"})

    assert "7006652" in result


def test_calculator_rejects_non_numeric_expression():
    result = calculator.invoke({"expression": "__import__('os').system('ls')"})

    assert "计算错误" in result


def test_python_executor_runs_simple_code():
    result = python_executor.invoke({"code": "print(sum([1, 2, 3]))"})

    assert "执行成功" in result
    assert "6" in result


def test_python_executor_blocks_file_access():
    result = python_executor.invoke({"code": "open('/tmp/x', 'w').write('bad')"})

    assert "安全限制" in result
    assert "open(" in result


def test_builtin_tools_do_not_hardcode_mcp_weather_tool():
    names = {tool.name for tool in get_builtin_tools()}

    assert "weather_query" not in names


def test_datetime_uses_requested_timezone():
    result = get_datetime.invoke({"timezone": "Asia/Shanghai"})

    assert "当前时间" in result
    assert "Asia/Shanghai" in result


def test_web_search_without_key_degrades_cleanly(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(settings, "tavily_api_key", "")

    result = web_search.invoke({"query": "特斯拉最新股价"})

    assert "TAVILY_API_KEY" in result


def test_web_search_replaces_garbled_result_content(monkeypatch):
    class FakeTavilyClient:
        def __init__(self, api_key: str):
            self.api_key = api_key

        def search(self, query: str, max_results: int):
            return {
                "results": [
                    {
                        "title": "深证成份指数(399001)_最新成分 - 新浪",
                        "content": "؛0000.000.0000.00֒|0000.000.0000.00 ݖ֒|300: 0000.000.0000.00Ԟ.# 301275 2025-12-15",
                        "url": "http://vip.stock.finance.sina.com.cn/corp/go.php/vII_NewestComponent/indexid/399001.phtml",
                    }
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "tavily", types.SimpleNamespace(TavilyClient=FakeTavilyClient))

    result = web_search.invoke({"query": "深证成指最新成分", "max_results": 1})

    assert "标题: 深证成份指数(399001)_最新成分 - 新浪" in result
    assert "摘要不可用" in result
    assert "0000.000.0000" not in result


def test_wikipedia_without_dependency_degrades_cleanly():
    result = wikipedia_search.invoke({"query": "量子纠缠"})

    assert result
