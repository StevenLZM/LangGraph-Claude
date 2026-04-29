from __future__ import annotations

from tools.builtin import calculator, get_datetime, python_executor, weather_query, web_search, wikipedia_search


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


def test_weather_query_uses_internal_city_data():
    result = weather_query.invoke({"city": "北京"})

    assert "北京" in result
    assert "温度" in result
    assert "数据源: internal-mcp" in result


def test_datetime_uses_requested_timezone():
    result = get_datetime.invoke({"timezone": "Asia/Shanghai"})

    assert "当前时间" in result
    assert "Asia/Shanghai" in result


def test_web_search_without_key_degrades_cleanly(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = web_search.invoke({"query": "特斯拉最新股价"})

    assert "TAVILY_API_KEY" in result


def test_wikipedia_without_dependency_degrades_cleanly():
    result = wikipedia_search.invoke({"query": "量子纠缠"})

    assert result
