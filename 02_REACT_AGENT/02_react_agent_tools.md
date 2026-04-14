# PRD：ReAct Agent — 智能工具调用助手

> 项目编号：02 | 难度：⭐⭐⭐ | 预计周期：2 周

---

## 一、项目背景与目标

### 背景

纯 LLM 只能基于训练数据回答问题，无法获取实时信息（天气、股价、新闻），也无法执行真实操作（计算、代码运行、文件读写）。ReAct（Reasoning + Acting）模式让 Agent 具备"思考 → 行动 → 观察 → 再思考"的循环能力。

### 目标

构建一个具备多种工具调用能力的 ReAct Agent，能够自主规划并调用工具完成复杂任务，无需人工指定调用哪个工具。

### 学习目标

| 技能点 | 掌握内容 |
|--------|----------|
| ReAct 模式 | Thought/Action/Observation 循环原理 |
| Function Calling | LLM 工具调用协议与参数定义 |
| 自定义工具 | LangChain @tool 装饰器封装 |
| Agent 执行器 | AgentExecutor 配置与调试 |
| 错误处理 | 工具失败时的重试与降级逻辑 |
| 流式输出 | 实时展示 Agent 思考过程 |

---

## 二、用户故事

```
作为一名研究人员
我想通过自然语言向 AI 助手提问
它能自动决定是否搜索网络、执行计算或运行代码
从而给我一个综合、准确的回答
而不是让我手动去各个工具查找再整合
```

---

## 三、功能需求

### 3.1 内置工具集

- **T01 网络搜索**：接入 Tavily / SerpAPI，获取实时信息
- **T02 计算器**：执行数学表达式，避免 LLM 计算错误
- **T03 代码执行**：在沙箱中运行 Python 代码并返回结果
- **T04 天气查询**：调用 OpenWeatherMap API 获取天气
- **T05 日期时间**：获取当前日期、时间及时区转换
- **T06 维基百科**：查询词条摘要信息

### 3.2 Agent 行为

- **F01** 自动分析用户意图，决定调用哪个（或哪些）工具
- **F02** 实时流式展示推理链：`思考 → 调用工具 → 观察结果 → 继续思考`
- **F03** 单次任务最多允许 10 步工具调用，防止死循环
- **F04** 工具调用失败时，自动重试 1 次或切换备用方案
- **F05** 最终答案综合所有工具结果，给出清晰回复

### 3.3 交互界面

- **F06** 聊天界面，支持多轮对话
- **F07** 侧边栏展示本次对话的完整推理链（可折叠）
- **F08** 每次工具调用显示：工具名、输入参数、返回结果

---

## 四、非功能需求

| 指标 | 要求 |
|------|------|
| 工具调用延迟 | 单次工具调用 ≤ 5 秒 |
| 总响应时间 | 复杂任务（≤5步）≤ 30 秒 |
| 安全性 | 代码执行在隔离沙箱，禁止访问文件系统和网络 |
| 可观测性 | 完整记录每步推理链，便于调试 |

---

## 五、技术架构

```
用户输入
    │
    ▼
Agent 规划层 (ReAct Prompt + LLM)
    │
    ├── 判断是否需要工具
    │       │
    │       ▼
    │   工具路由层
    │   ┌─────────────────────────────────┐
    │   │  搜索 │ 计算 │ 代码 │ 天气 │ ... │
    │   └─────────────────────────────────┘
    │       │
    │       ▼
    │   工具执行 → 返回 Observation
    │       │
    └── 继续推理 or 输出最终答案
    │
    ▼
Streamlit 界面（流式输出推理过程）
```

### 技术选型

| 层次 | 技术 |
|------|------|
| 框架 | LangChain AgentExecutor |
| 工具定义 | `@tool` 装饰器 + Pydantic 参数校验 |
| 搜索 | Tavily Search API |
| 代码执行 | RestrictedPython / Docker 沙箱 |
| LLM | claude-sonnet​-4-6 / GPT-4o（支持 Function Calling）|
| UI | Streamlit with `st.status` 流式展示 |

---

## 六、核心实现要点

### 6.1 自定义工具定义

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词，应简洁明确")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """当需要获取实时信息、新闻、当前事件时使用此工具"""
    result = tavily_client.search(query)
    return result["answer"]
```

### 6.2 Agent 构建

```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt  # 包含 Thought/Action/Observation 格式
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,
    handle_parsing_errors=True,  # 容错处理
    verbose=True
)
```

### 6.3 流式输出推理链

```python
for chunk in executor.stream({"input": user_query}):
    if "actions" in chunk:
        # 展示工具调用
        action = chunk["actions"][0]
        st.write(f"🔧 调用工具: {action.tool}")
        st.write(f"📥 参数: {action.tool_input}")
    elif "observations" in chunk:
        # 展示工具返回
        st.write(f"📤 结果: {chunk['observations'][0]}")
    elif "output" in chunk:
        # 最终答案
        st.write(f"✅ 最终答案: {chunk['output']}")
```

---

## 七、典型测试用例

| 用户输入 | 期望行为 |
|----------|----------|
| "今天北京天气怎么样，适合跑步吗？" | 调用天气工具 → 综合判断 |
| "1234 * 5678 等于多少？" | 调用计算器，不用 LLM 直接算 |
| "写一个冒泡排序并运行测试" | 调用代码执行工具 |
| "特斯拉最新股价是多少？" | 调用搜索工具获取实时数据 |
| "今天是星期几？距离春节还有多少天？" | 调用日期工具 + 计算器 |

---

## 八、评估标准

- [ ] 6 个工具均可正常调用并返回正确结果
- [ ] Agent 能自动选择正确工具，无需用户提示
- [ ] 推理链在界面上完整可见
- [ ] 工具调用失败时不崩溃，给出友好提示
- [ ] 恶意代码执行请求被沙箱拦截

---

## 九、项目交付物

1. `app.py` — Streamlit 主程序
2. `agent/tools.py` — 所有工具定义
3. `agent/executor.py` — Agent 初始化与执行
4. `agent/prompts.py` — ReAct Prompt 模板
5. `README.md` — 工具列表说明与演示 GIF
6. `tests/test_tools.py` — 各工具单元测试

---

## 十、扩展方向

- 添加文件读写工具（读 CSV、写报告）
- 添加数据库查询工具（Text-to-SQL）
- 接入 MCP（Model Context Protocol）标准化工具接口
- 实现工具调用的异步并发（多个工具同时执行）
