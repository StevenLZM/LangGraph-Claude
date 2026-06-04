# ReAct Agent 智能工具助手

本工程基于 `02_react_agent_tools.md` 和 `02_react_agent_engineering.md` 落地，用一个 Streamlit 应用演示两种 Agent 设计：

- `ReAct`：LLM 在「思考 -> 工具调用 -> 观察」循环中自主选择工具。
- `Plan-and-Execute`：先生成结构化计划，再按步骤执行并汇总答案。

## 运行

```bash
cd 02_REACT_AGENT
python -m pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

`.env` 中至少需要配置 `DEEPSEEK_API_KEY`。`TAVILY_API_KEY` 是可选项，未配置时搜索工具会给出降级提示。

## 工具

默认工具集由 `tools/registry.py` 统一加载：本地内置工具来自 `tools/builtin.py`，MCP 工具从 `.mcp.json` 动态发现并转换为 LangChain tool。

- `web_search`：实时互联网搜索，依赖 Tavily。
- `calculator`：安全 AST 数学计算器。
- `python_executor`：受限 Python 代码执行器，禁止导入、文件系统和网络访问。
- `get_datetime`：指定时区日期时间。
- `wikipedia_search`：维基百科摘要查询。
- `weather_query`：由 `.mcp.json` 启动 `mcp_servers.weather_server` 后动态加载。

## 测试

```bash
cd 02_REACT_AGENT
pytest tests -q
```
