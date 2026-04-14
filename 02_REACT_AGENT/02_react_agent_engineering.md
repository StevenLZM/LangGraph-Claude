# 工程设计：ReAct Agent — 智能工具调用助手

> 对应 PRD：02_react_agent_tools.md

---

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  [用户输入]  [实时推理链展示]  [工具调用详情]  [最终答案]         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ invoke / stream
┌───────────────────────────▼─────────────────────────────────────┐
│                     ReAct Agent 核心                             │
│                                                                 │
│   用户输入 → [LLM 思考] → [选择工具] → [执行工具] → [观察结果]  │
│                  ↑_______________________________________________│
│                           循环直到得出最终答案                   │
└─────────────────────────────────────────────────────────────────┘
                            │ tool calls
┌───────────────────────────▼─────────────────────────────────────┐
│                       工具执行层                                 │
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ 网络搜索  │ │  计算器  │ │ 代码执行  │ │  MCP Tool Server │   │
│  │ (Tavily) │ │ (sympy)  │ │(sandbox) │ │ (天气/Wiki/日期) │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、ReAct 执行流程详解

### 2.1 完整推理循环

```
用户: "今天北京天气怎么样？适合跑步吗？"
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Thought 1:                                         │
│  "用户想知道北京当前天气和是否适合跑步。             │
│   我需要先获取实时天气数据，再综合判断。             │
│   使用 weather_query 工具。"                        │
└──────────────────────────┬──────────────────────────┘
                           │ Action
                           ▼
                  tool: weather_query
                  input: {"city": "北京"}
                           │
                           ▼
                  Observation: {
                    "temp": 18, "condition": "晴",
                    "humidity": 45, "wind": "微风"
                  }
                           │
┌──────────────────────────▼──────────────────────────┐
│  Thought 2:                                         │
│  "北京今天18℃，晴天，湿度45%，微风。               │
│   这是非常适合户外运动的条件。                       │
│   我已有足够信息，可以给出最终答案。"               │
└──────────────────────────┬──────────────────────────┘
                           │ Final Answer
                           ▼
  "今天北京天气晴朗，气温18℃，湿度适中（45%），
   微风，非常适合跑步！建议穿薄外套，避开正午时段。"
```

### 2.2 多工具协作流程（复杂任务）

```
用户: "帮我查一下特斯拉最新市值，并计算它是苹果市值的百分之几"
      │
      ├─ Thought 1 → Action: web_search("特斯拉最新市值 2026")
      │              Observation: "约8000亿美元"
      │
      ├─ Thought 2 → Action: web_search("苹果公司最新市值 2026")
      │              Observation: "约3.3万亿美元"
      │
      ├─ Thought 3 → Action: calculator("8000 / 33000 * 100")
      │              Observation: "24.24"
      │
      └─ Thought 4 → Final Answer: "特斯拉市值约8000亿美元，
                     苹果约3.3万亿美元，特斯拉约为苹果的24.2%"
```

---

## 三、MCP 集成设计

### 3.1 MCP Server 配置

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["-m", "mcp_servers.weather_server"],
      "env": {
        "OPENWEATHER_API_KEY": "${OPENWEATHER_API_KEY}"
      },
      "description": "天气查询 MCP Server"
    },
    "wikipedia": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-wikipedia"],
      "description": "维基百科词条查询"
    },
    "datetime": {
      "command": "python",
      "args": ["-m", "mcp_servers.datetime_server"],
      "description": "日期时间工具"
    }
  }
}
```

### 3.2 自定义天气 MCP Server 实现

```python
# mcp_servers/weather_server.py
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("weather-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的当前天气信息，包括温度、湿度、天气状况",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，支持中文（如：北京、上海）"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "温度单位，metric=摄氏度"
                    }
                },
                "required": ["city"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_weather":
        city = arguments["city"]
        units = arguments.get("units", "metric")
        weather_data = await fetch_openweather(city, units)
        return [TextContent(
            type="text",
            text=f"城市: {city}\n温度: {weather_data['temp']}°C\n"
                 f"天气: {weather_data['condition']}\n"
                 f"湿度: {weather_data['humidity']}%\n"
                 f"风速: {weather_data['wind_speed']}m/s"
        )]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                        InitializationOptions(server_name="weather"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3.3 MCP 工具转换为 LangChain Tool

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def load_mcp_as_langchain_tools():
    """将 MCP Server 的工具加载为 LangChain 工具"""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "mcp_servers.weather_server"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 自动将 MCP tools 转换为 LangChain BaseTool 格式
            tools = await load_mcp_tools(session)
            return tools
```

---

## 四、工具设计详情

### 4.1 工具清单与实现

```python
# agent/tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import subprocess, ast, operator, re

# ── Tool 1: 网络搜索 ──────────────────────────────────────
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词，应简洁明确，建议5-20个字")
    max_results: int = Field(default=3, ge=1, le=5, description="返回结果数量")

@tool("web_search", args_schema=SearchInput, return_direct=False)
def web_search(query: str, max_results: int = 3) -> str:
    """搜索互联网获取实时信息。适用于：新闻、当前事件、实时数据、
    最新价格、近期发布的内容。不适用于：数学计算、固定知识。"""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = client.search(query=query, max_results=max_results)
    results = []
    for r in response["results"]:
        results.append(f"标题: {r['title']}\n内容: {r['content'][:300]}\n来源: {r['url']}")
    return "\n\n---\n\n".join(results)

# ── Tool 2: 数学计算器 ────────────────────────────────────
class CalcInput(BaseModel):
    expression: str = Field(description="数学表达式，如: 1234 * 5678 或 sqrt(144)")

@tool("calculator", args_schema=CalcInput)
def calculator(expression: str) -> str:
    """执行精确的数学计算。适用于：四则运算、百分比、开方、
    对数等。注意：不能处理变量，只能是纯数字表达式。"""
    import sympy
    try:
        result = sympy.sympify(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# ── Tool 3: Python 代码执行 ────────────────────────────────
class CodeInput(BaseModel):
    code: str = Field(description="要执行的 Python 代码，必须是完整可运行的代码片段")
    timeout: int = Field(default=10, le=30, description="最大执行时间（秒）")

@tool("python_executor", args_schema=CodeInput)
def python_executor(code: str, timeout: int = 10) -> str:
    """在安全沙箱中执行 Python 代码并返回输出。
    适用于：数据处理、算法验证、复杂计算。
    限制：无法访问网络和文件系统，不能安装包。"""
    # 安全检查
    forbidden = ["import os", "import sys", "open(", "subprocess", "__import__"]
    for f in forbidden:
        if f in code:
            return f"安全限制: 禁止使用 '{f}'"
    
    # 使用 RestrictedPython 或隔离进程执行
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True,
            timeout=timeout,
            # 限制资源
            env={"PATH": "/usr/bin"}
        )
        if result.returncode == 0:
            return f"执行成功:\n{result.stdout}"
        else:
            return f"执行错误:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return f"执行超时（{timeout}秒）"

# ── Tool 4: 维基百科 ──────────────────────────────────────
@tool("wikipedia_search")
def wikipedia_search(query: str) -> str:
    """查询维基百科获取概念解释和背景知识。
    适用于：人物、地点、历史事件、科学概念。
    不适用于：实时信息、当前事件。"""
    import wikipedia
    wikipedia.set_lang("zh")
    try:
        page = wikipedia.page(query)
        return page.summary[:1000]
    except wikipedia.DisambiguationError as e:
        return f"该词有多个含义，请更具体: {e.options[:5]}"
    except wikipedia.PageError:
        return "未找到相关词条"

# ── Tool 5: 日期时间 ──────────────────────────────────────
@tool("get_datetime")
def get_datetime(timezone: str = "Asia/Shanghai") -> str:
    """获取当前日期和时间。
    适用于：需要知道今天是几号、星期几、距某日还有多少天。"""
    from datetime import datetime
    import pytz
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    return (f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
            f"星期: {['一','二','三','四','五','六','日'][now.weekday()]}\n"
            f"时区: {timezone}")
```

### 4.2 工具错误处理装饰器

```python
import functools
from tenacity import retry, stop_after_attempt, wait_exponential

def tool_with_retry(max_attempts=2):
    """工具调用重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        return f"工具执行失败（已重试{max_attempts}次）: {str(e)}"
                    time.sleep(1)
        return wrapper
    return decorator
```

---

## 五、Agent 构建设计

### 5.1 使用 LangGraph 重构 ReAct（推荐）

```python
# agent/graph.py - 用 LangGraph 实现 ReAct，更灵活可控
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int     # 防死循环计数
    tool_calls_log: list     # 工具调用记录

# 定义所有工具
all_tools = [web_search, calculator, python_executor, wikipedia_search, get_datetime]

# LLM 绑定工具
llm_with_tools = llm.bind_tools(all_tools)

def agent_node(state: AgentState):
    """LLM 推理节点：分析当前状态，决定下一步行动"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1
    }

def should_continue(state: AgentState):
    """路由函数：判断继续调用工具还是结束"""
    last_message = state["messages"][-1]
    
    # 超过最大迭代次数，强制结束
    if state["iteration_count"] >= 10:
        return "end"
    
    # 有工具调用则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 无工具调用则结束
    return "end"

# 构建图
tool_node = ToolNode(all_tools)  # 自动路由到对应工具

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")  # 工具执行后回到 agent

app = workflow.compile()
```

### 5.2 System Prompt（ReAct 风格）

```python
REACT_SYSTEM_PROMPT = """你是一个强大的 AI 助手，可以使用多种工具来完成用户任务。

【可用工具说明】
- web_search: 搜索实时信息（新闻、价格、当前事件）
- calculator: 精确数学计算（避免 LLM 直接计算的误差）
- python_executor: 执行 Python 代码验证算法
- wikipedia_search: 查询知识概念和背景信息
- get_datetime: 获取当前日期时间

【行为准则】
1. 优先思考是否需要工具，不要为了用工具而用工具
2. 对于数学计算，必须使用 calculator 而非直接计算
3. 对于实时信息，必须使用 web_search
4. 每次工具调用后，认真分析结果再决定下一步
5. 工具失败时，尝试换种方式或告知用户

【输出格式】
在没有工具调用时，直接给出清晰完整的最终答案。
引用工具结果时注明数据来源。
"""
```

---

## 六、流式输出设计

```python
# app.py - Streamlit 流式展示推理过程
import streamlit as st

def display_agent_stream(user_input: str):
    """实时展示 Agent 推理链"""
    
    with st.container():
        reasoning_placeholder = st.empty()
        reasoning_steps = []
        
        for event in app.stream(
            {"messages": [("user", user_input)], "iteration_count": 0},
            stream_mode="updates"
        ):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # 展示工具调用
                        for tc in msg.tool_calls:
                            step = {
                                "type": "tool_call",
                                "tool": tc["name"],
                                "input": tc["args"]
                            }
                            reasoning_steps.append(step)
                    
                    elif hasattr(msg, "content") and node_name == "tools":
                        # 展示工具结果
                        reasoning_steps[-1]["output"] = msg.content
                    
                    elif hasattr(msg, "content") and node_name == "agent":
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            # 最终答案
                            reasoning_steps.append({
                                "type": "final_answer",
                                "content": msg.content
                            })
                
                # 实时渲染
                with reasoning_placeholder.container():
                    for i, step in enumerate(reasoning_steps):
                        if step["type"] == "tool_call":
                            with st.expander(f"步骤 {i+1}: 调用 {step['tool']}", expanded=True):
                                st.code(str(step["input"]), language="json")
                                if "output" in step:
                                    st.text(step["output"][:500])
                        elif step["type"] == "final_answer":
                            st.success(step["content"])
```

---

## 七、目录结构

```
02_react_agent/
├── app.py                      # Streamlit 主程序
├── agent/
│   ├── graph.py                # LangGraph ReAct 图
│   ├── tools.py                # 所有工具定义
│   └── prompts.py              # System Prompt
├── mcp_servers/
│   ├── weather_server.py       # 天气 MCP Server
│   └── datetime_server.py      # 日期时间 MCP Server
├── mcp/
│   └── loader.py               # MCP 工具加载器
├── sandbox/
│   └── executor.py             # 代码执行沙箱
├── .mcp.json                   # MCP 配置
├── .env.example
└── requirements.txt
```

---

## 八、测试矩阵

| 测试场景 | 期望工具 | 期望行为 |
|----------|----------|----------|
| "1234 * 5678" | calculator | 不用 LLM 直接算，调用计算器 |
| "今天北京天气" | weather MCP | 调用天气工具，非搜索 |
| "特斯拉股价" | web_search | 明确是实时数据，调用搜索 |
| "写个快排并运行" | python_executor | 生成代码并在沙箱运行 |
| "量子纠缠是什么" | wikipedia_search | 固定知识，优先 Wiki |
| "今天是星期几" | get_datetime | 不猜测，调用日期工具 |
| "你好" | 无 | 直接回答，不调用工具 |
