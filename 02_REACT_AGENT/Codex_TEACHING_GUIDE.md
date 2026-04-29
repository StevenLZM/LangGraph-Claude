# 02 ReAct Agent 教学文档

这份文档用于讲解 `02_REACT_AGENT` 工程。目标不是逐行背代码，而是让你能用代码解释清楚三个面试核心问题：

1. Agent 如何决定是否调用工具。
2. ReAct 和 Plan-and-Execute 的工程差异。
3. 工具、MCP、LLM 配置、UI、测试如何组成一个可演示项目。

---

## 1. 项目一句话介绍

`02_REACT_AGENT` 是一个 Streamlit 工具调用助手。它支持两种 Agent 模式：

- `ReAct`：边推理、边调用工具、边观察结果，适合短链路工具调用。
- `Plan-and-Execute`：先生成计划，再逐步执行，适合多步骤任务。

核心目录：

```text
02_REACT_AGENT/
├── app.py                    # Streamlit UI，模式切换和推理链展示
├── agent/
│   ├── react.py              # LangGraph ReAct 循环
│   ├── plan_execute.py       # Plan-and-Execute 图
│   ├── events.py             # UI 事件模型
│   └── prompts.py            # 系统提示词
├── tools/builtin.py          # 6 个 LangChain 工具
├── mcp_servers/weather_*     # 内部天气 MCP server 和模拟数据
├── sandbox/executor.py       # 受限 Python 执行器
├── config/llm.py             # DeepSeek LLM 工厂
└── tests/                    # 可离线运行的单元测试
```

面试讲法：

> 这个项目把 LLM 工具调用拆成三个层次：LLM 负责决策，LangGraph 负责编排循环，工具层负责真实执行。ReAct 和 Plan-and-Execute 共用工具和 UI 事件模型，所以可以在同一个应用中对比两种 Agent 架构。

---

## 2. DeepSeek LLM 配置

文件：`config/llm.py`

关键代码：

```python
@lru_cache(maxsize=8)
def get_llm(tier: Tier = "max", *, temperature: float = 0.2, streaming: bool = False) -> ChatOpenAI:
    if not settings.deepseek_api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置。请在 02_REACT_AGENT/.env 中填入。")

    model = settings.deepseek_max_model if tier == "max" else settings.deepseek_light_model
    return ChatOpenAI(
        model=model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=temperature,
        streaming=streaming,
        timeout=120,
        max_retries=2,
    )
```

讲解重点：

- DeepSeek 提供 OpenAI-compatible API，所以这里复用 `langchain_openai.ChatOpenAI`。
- `tier="max"` 用于规划、最终推理等更重的步骤。
- `tier="turbo"` 用于执行子步骤，降低成本。
- `@lru_cache` 避免重复创建 LLM client。
- 缺少 `DEEPSEEK_API_KEY` 时主动抛出清晰错误，方便排查配置问题。

面试重点：

> 如果模型厂商提供 OpenAI 兼容接口，通常不需要重写 LangChain 集成，只需要替换 `base_url`、`api_key` 和 `model`。这降低了供应商迁移成本。

---

## 3. 工具层：让 LLM 能“行动”

文件：`tools/builtin.py`

本项目实现了 6 个工具：

```python
def get_tools():
    return [web_search, calculator, python_executor, weather_query, get_datetime, wikipedia_search]
```

每个工具都用 LangChain 的 `@tool` 包装，并用 Pydantic schema 定义参数。例如计算器：

```python
class CalcInput(BaseModel):
    expression: str = Field(description="数学表达式，如 1234 * 5678 或 sqrt(144)")


@tool("calculator", args_schema=CalcInput)
def calculator(expression: str) -> str:
    """执行精确数学计算。数学问题必须优先使用此工具。"""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
    except Exception as exc:
        return f"计算错误: {exc}"
    return f"计算结果: {expression} = {result}"
```

为什么不用 `eval`：

- `eval("__import__('os').system('rm -rf /')")` 这类表达式有安全风险。
- 当前实现用 `ast.parse` 解析语法树，再只允许数字、四则运算、幂、取模和少量数学函数。
- 面试中可以强调：工具执行必须有输入约束和安全边界，不能直接信任 LLM 生成内容。

搜索工具的降级设计：

```python
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    return "搜索工具未配置：请设置 TAVILY_API_KEY 后再查询实时互联网信息。"
```

这体现了工程实践中的 graceful degradation：

- 配置缺失不让应用崩溃。
- 明确告诉用户缺什么。
- 测试可以离线运行，不依赖真实网络。

面试重点：

> 工具不是简单函数。一个可用于 Agent 的工具需要有清晰名称、自然语言描述、参数 schema、错误处理和安全边界。LLM 是根据工具描述和 schema 决定是否调用工具的。

---

## 4. Python 沙箱：演示级安全边界

文件：`sandbox/executor.py`

关键代码：

```python
FORBIDDEN_SNIPPETS = (
    "import ",
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "subprocess",
    "socket",
    "requests",
)

SAFE_BUILTINS = {
    "print": print,
    "range": range,
    "sum": sum,
    "len": len,
    "sorted": sorted,
}
```

执行逻辑：

```python
def run_python_code(code: str) -> str:
    for forbidden in FORBIDDEN_SNIPPETS:
        if forbidden in code:
            return f"安全限制: 禁止使用 '{forbidden}'"

    stdout = io.StringIO()
    globals_dict = {"__builtins__": SAFE_BUILTINS}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, globals_dict, {})
    except Exception as exc:
        return f"执行错误:\n{type(exc).__name__}: {exc}"
```

讲解重点：

- 这是教学演示级沙箱，不是生产级隔离。
- 它限制导入、文件系统、网络、动态执行。
- 真实生产场景更推荐 Docker、Firecracker、gVisor、进程资源限制、网络隔离和超时控制。

面试重点：

> LLM 生成代码不能直接在宿主环境执行。哪怕是 Demo，也要明确安全边界和生产改造方向。

---

## 5. 内部天气 MCP

文件：

- `mcp_servers/weather_server.py`
- `mcp_servers/weather_data.py`
- `.mcp.json`

MCP server 暴露一个工具：

```python
Tool(
    name="weather_query",
    description="查询内部天气 MCP 数据，适合回答城市天气和户外活动建议。",
    inputSchema=WEATHER_QUERY_SCHEMA,
)
```

调用工具时返回 JSON 文本：

```python
@app.call_tool()
async def _call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "weather_query":
        raise ValueError(f"未知 tool: {name}")
    data = await weather_data.get_weather(
        city=str(arguments["city"]),
        units=str(arguments.get("units", "metric")),
    )
    payload = {"data": data, "text": weather_data.format_weather(data)}
    return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
```

为什么天气用内部 MCP：

- 天气是典型外部工具能力，适合演示 MCP 的“工具 server”角色。
- 使用本地模拟数据，不依赖 OpenWeather API key，方便教学、测试和演示。
- `.mcp.json` 保留了 MCP server 配置形式：

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["-m", "mcp_servers.weather_server"]
    }
  }
}
```

面试重点：

> MCP 的价值是把工具能力标准化成独立 server。Agent 不需要知道工具内部怎么实现，只要知道工具 schema 和调用协议。

---

## 6. ReAct 实现：推理和行动交替循环

文件：`agent/react.py`

状态定义：

```python
class ReactState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration_count: int
```

解释：

- `messages` 保存用户消息、AI 消息、工具结果。
- `add_messages` 是 LangGraph reducer，节点返回的新消息会追加到已有消息里。
- `iteration_count` 防止 Agent 死循环。

LLM 节点：

```python
def agent_node(state: ReactState) -> ReactState:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=REACT_SYSTEM_PROMPT), *messages]
    response = bound_llm.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
    }
```

路由逻辑：

```python
def should_continue(state: ReactState) -> str:
    if int(state.get("iteration_count", 0)) >= max_iters:
        return "end"
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else "end"
```

图结构：

```python
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tool_list))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")
```

运行过程：

```text
用户输入
  -> agent 节点调用 LLM
  -> 如果 AIMessage 中有 tool_calls，进入 tools 节点
  -> ToolNode 执行工具并生成 ToolMessage
  -> 回到 agent 节点继续推理
  -> 没有 tool_calls 时结束
```

面试重点：

> ReAct 的本质是闭环：Reasoning 决定 Action，Action 产生 Observation，Observation 再进入下一轮 Reasoning。工程上要额外处理最大迭代次数、工具失败、消息累积和可观测性。

---

## 7. Plan-and-Execute 实现：先规划再执行

文件：`agent/plan_execute.py`

结构化计划 schema：

```python
class PlanStep(BaseModel):
    id: int = Field(ge=1)
    objective: str
    suggested_tool: str = "none"


class Plan(BaseModel):
    steps: list[PlanStep]
```

规划器：

```python
def default_planner(user_input: str) -> Plan:
    llm = get_llm("max", temperature=0.1)
    structured = llm.with_structured_output(Plan, method="json_mode")
    result = structured.invoke(
        [
            SystemMessage(content=PLAN_SYSTEM_PROMPT),
            HumanMessage(content=f"用户任务：{user_input}\n请输出 JSON。"),
        ]
    )
    if isinstance(result, Plan):
        return result
    return Plan.model_validate(result)
```

执行器复用 ReAct：

```python
def default_executor(step: PlanStep, user_input: str, past_steps: list[StepResult]) -> StepResult:
    history = "\n".join(f"{item.step_id}. {item.objective}: {item.output}" for item in past_steps) or "无"
    prompt = (
        f"原始任务：{user_input}\n"
        f"已完成步骤：\n{history}\n\n"
        f"当前步骤：{step.objective}\n"
        f"建议工具：{step.suggested_tool}\n"
        "请完成当前步骤；如果需要工具，自动调用合适工具。"
    )
    result = run_react(prompt, llm=get_llm("turbo", temperature=0.1))
    return StepResult(...)
```

图结构：

```python
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("finalizer", finalizer_node)
workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", route_after_plan, {"executor": "executor", "finalizer": "finalizer"})
workflow.add_conditional_edges("executor", route_after_execute, {"executor": "executor", "finalizer": "finalizer"})
workflow.add_edge("finalizer", END)
```

运行过程：

```text
用户输入
  -> planner 生成 Plan
  -> executor 执行当前 step
  -> 如果还有 step，继续 executor
  -> finalizer 汇总步骤结果
```

面试重点：

> Plan-and-Execute 把“任务拆解”和“工具执行”拆开了。优点是复杂任务更可控、更容易展示计划；缺点是如果初始计划错了，后续步骤会被带偏，所以生产系统常加入 replan、人工确认或反思节点。

---

## 8. ReAct vs Plan-and-Execute 对比

| 维度 | ReAct | Plan-and-Execute |
|---|---|---|
| 决策方式 | 每轮根据上下文决定下一步 | 先生成完整计划，再逐步执行 |
| 适合场景 | 查询天气、计算、短任务、多工具少步骤 | 研究、报告、排查问题、长链路任务 |
| 优点 | 灵活、响应快、实现简单 | 结构清晰、可展示计划、便于审计 |
| 风险 | 容易循环，需要 max_iterations | 初始计划错误会影响后续执行 |
| 本项目实现 | `agent/react.py` | `agent/plan_execute.py` |

面试回答模板：

> ReAct 适合不确定下一步的即时工具调用，它把推理和行动交替进行。Plan-and-Execute 适合长任务，先把目标拆成结构化步骤，再逐步执行。两者不是互斥关系，本项目里 Plan-and-Execute 的 executor 仍然复用了 ReAct，说明复杂 Agent 往往是多种模式组合。

---

## 9. UI 和可观测性

文件：

- `agent/events.py`
- `app.py`

事件模型：

```python
@dataclass(slots=True)
class AgentEvent:
    type: EventType
    title: str
    content: str = ""
    tool: str | None = None
    tool_input: Any = None
    tool_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

为什么要有事件模型：

- UI 不直接解析 LangChain message 细节。
- ReAct 和 Plan-and-Execute 可以共用展示层。
- 每个工具调用、工具结果、计划步骤、最终答案都能被记录和展示。

Streamlit 模式切换：

```python
mode = st.radio("执行模式", ["ReAct", "Plan-and-Execute"], horizontal=False)
```

运行分发：

```python
def _run(mode: str, prompt: str):
    if mode == "ReAct":
        return run_react(prompt)
    return run_plan_and_execute(prompt)
```

面试重点：

> Agent 系统不能只返回最终答案。面试时要强调可观测性：工具调用输入、工具输出、中间步骤、最终答案都应该可追踪，否则很难调试幻觉、错误工具调用和循环问题。

---

## 10. 测试怎么讲

测试目录：`tests/`

本项目测试分四类：

```text
test_deepseek_config.py       # DeepSeek 配置和模型档位
test_tools.py                 # 工具行为和降级
test_weather_mcp.py           # 内部 MCP server
test_react_graph.py           # ReAct 图循环
test_plan_execute_graph.py    # Plan-and-Execute 图
```

ReAct 测试使用 fake LLM，不调用真实 DeepSeek：

```python
class _ToolCallingLLM:
    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        ...
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "calculator",
                    "args": {"expression": "1234 * 5678"},
                    "id": "calc-1",
                }
            ],
        )
```

测试断言：

```python
assert any(isinstance(message, ToolMessage) for message in state["messages"])
assert state["messages"][-1].content == "1234 * 5678 = 7006652"
assert state["iteration_count"] == 2
```

面试重点：

> Agent 测试不要依赖真实 LLM 稳定输出。核心逻辑应该用 fake LLM 固定 tool_calls，测试图编排、工具执行、事件生成是否正确。真实 LLM 调用可以放到少量集成测试或人工演示中。

---

## 11. 推荐讲解顺序

演示时按这个顺序讲，最清晰：

1. 打开 `README.md`，说明项目目标和两种模式。
2. 打开 `config/llm.py`，说明 DeepSeek 兼容 OpenAI API 的接入方式。
3. 打开 `tools/builtin.py`，讲 6 个工具和 `@tool + args_schema`。
4. 打开 `mcp_servers/weather_server.py`，讲内部天气 MCP。
5. 打开 `agent/react.py`，画出 `agent -> tools -> agent` 循环。
6. 打开 `agent/plan_execute.py`，讲 `planner -> executor -> finalizer`。
7. 打开 `app.py`，讲 UI 如何展示推理链。
8. 打开 `tests/`，讲为什么用 fake LLM 测试 Agent。

---

## 12. 高频面试题与回答要点

### Q1：什么是 ReAct？

回答要点：

- ReAct = Reasoning + Acting。
- LLM 先推理是否需要工具，再产生 tool call。
- 工具执行结果作为 Observation 回到上下文。
- 循环直到 LLM 不再调用工具，输出最终答案。
- 工程上要限制最大迭代次数，避免死循环。

结合本项目：

> `agent/react.py` 中 `should_continue()` 判断最后一条 AIMessage 是否有 `tool_calls`，有就进入 `ToolNode`，没有就结束。

### Q2：为什么使用 LangGraph，而不是普通函数循环？

回答要点：

- LangGraph 把流程显式建模为节点和边。
- 状态由 `TypedDict` 管理，消息用 reducer 累积。
- 条件路由清晰，便于扩展成更多节点。
- 更接近生产 Agent 的状态机模型。

结合本项目：

> ReAct 图只有两个节点，但已经体现了 LangGraph 的核心抽象：`agent` 节点负责 LLM 决策，`tools` 节点负责执行工具，条件边决定继续还是结束。

### Q3：Function Calling 和工具 schema 为什么重要？

回答要点：

- LLM 不是直接调用 Python 函数，而是生成结构化 tool call。
- 工具名称、描述、参数 schema 会影响模型是否正确选择工具。
- Pydantic schema 能校验参数，减少脏输入。

结合本项目：

> `calculator` 的 `CalcInput` 告诉模型 `expression` 应该是数学表达式；`weather_query` 的 `WeatherInput` 告诉模型需要 `city`。

### Q4：MCP 在项目中解决什么问题？

回答要点：

- MCP 把工具能力封装成标准 server。
- Agent 可以通过协议发现工具和调用工具。
- 工具实现和 Agent 编排解耦。

结合本项目：

> 天气查询通过 `mcp_servers/weather_server.py` 暴露，虽然数据是内部模拟的，但接口形态符合 MCP server，可以替换成真实天气服务。

### Q5：ReAct 和 Plan-and-Execute 怎么选？

回答要点：

- 简单、短链路、需要即时工具判断：选 ReAct。
- 长任务、多步骤、需要展示计划和审计：选 Plan-and-Execute。
- 两者可以组合，Plan 的每一步也可以用 ReAct 执行。

结合本项目：

> `default_executor()` 就是 Plan-and-Execute 复用 ReAct 的例子。

### Q6：这个项目的安全风险有哪些？

回答要点：

- LLM 可能选择错误工具或构造危险工具输入。
- Python 代码执行有宿主机风险。
- 搜索结果可能被 prompt injection 污染。
- 工具返回内容不能无条件信任。

本项目已有措施：

- `python_executor` 禁止导入、文件、网络和动态执行。
- `calculator` 使用 AST 白名单。
- 工具缺配置时降级，不崩溃。

生产改造：

- 容器隔离代码执行。
- 工具权限分级和审批。
- 搜索结果清洗和引用约束。
- trace、日志、限流、超时和成本控制。

### Q7：如何测试 Agent？

回答要点：

- 单元测试不要依赖真实 LLM。
- 用 fake LLM 固定返回 `tool_calls`。
- 测试图编排、工具执行、事件生成。
- 少量集成测试验证真实模型和外部服务。

结合本项目：

> `tests/test_react_graph.py` 用 `_ToolCallingLLM` 模拟模型第一次调用 calculator，第二次返回最终答案。

### Q8：这个项目还能如何升级？

可讲方向：

- ReAct 增加异步 streaming 事件。
- Plan-and-Execute 增加 replan 节点和反思节点。
- MCP loader 真正从 `.mcp.json` 动态加载工具。
- 工具执行增加 timeout、retry、权限控制。
- 引入 LangSmith trace 做可观测性。
- 加入多轮记忆和 checkpoint。

---

## 13. 30 秒项目陈述

> 我做了一个基于 LangGraph 的工具调用 Agent 项目，支持 ReAct 和 Plan-and-Execute 两种模式。LLM 使用 DeepSeek 的 OpenAI-compatible API，工具层用 LangChain `@tool` 和 Pydantic schema 封装，包括搜索、计算器、代码执行、天气、日期和 Wikipedia。天气工具通过内部 MCP server 暴露，方便演示标准化工具协议。ReAct 部分是 `agent -> tools -> agent` 循环，Plan-and-Execute 部分先生成结构化计划，再逐步执行，每一步复用 ReAct。UI 用 Streamlit 展示工具调用和推理链，测试使用 fake LLM 离线验证图编排，避免依赖真实模型的不稳定输出。

---

## 14. 复习检查清单

- [ ] 能画出 ReAct 的 `agent -> tools -> agent` 循环。
- [ ] 能解释 `ToolNode` 的作用。
- [ ] 能说明为什么需要 `iteration_count`。
- [ ] 能解释 `@tool` 和 Pydantic args schema。
- [ ] 能讲清楚 DeepSeek 为什么可以用 `ChatOpenAI`。
- [ ] 能说明 MCP 和普通 Python 函数工具的差异。
- [ ] 能对比 ReAct 与 Plan-and-Execute。
- [ ] 能解释为什么测试里使用 fake LLM。
- [ ] 能指出当前代码执行沙箱的边界和生产改造方案。
