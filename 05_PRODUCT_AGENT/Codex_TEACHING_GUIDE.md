# 05 Product Agent 教学文档

这份文档用于讲解 `05_PRODUCT_AGENT` 当前已经开发完成的部分。重点不是背目录，而是能把一次客服请求从 UI 到 API、再到 LangGraph、规则决策、Mock 工具和测试完整讲清楚。

当前真实完成范围：

- M0：项目骨架、FastAPI 健康检查、基础配置、最小 LangGraph 图、pytest 测试。
- M1：`POST /chat`、内置客服 UI、Mock 订单/物流/商品/退款工具、规则型客服 Agent、转人工标记、退款二次确认、测试覆盖。

尚未完成范围：

- M2 记忆系统：短期窗口、会话恢复、长期用户记忆。
- M3 限流与弹性：Redis 限流、Token 预算、LLM fallback、熔断。
- M4 可观测性：Prometheus、LangSmith、质量评估。
- M5 Docker Compose 和压测。

面试时要明确：当前 M1 是“离线可测的客服 MVP”，不是完整生产级系统。

---

## 1. 项目一句话介绍

`05_PRODUCT_AGENT` 是一个生产级 AI 客服系统的渐进式实现。当前阶段先做出可演示、可测试的客服闭环：用户在 UI 输入问题，FastAPI 接收请求，LangGraph 组织状态流转，规则型客服决策调用 Mock 业务工具，最后返回回答、订单上下文、转人工状态和质量分占位。

面试讲法：

> 这个项目不是从一开始就接真实 LLM 和 Redis/Postgres，而是先把客服业务闭环、接口契约、状态结构、测试基线和 UI 演示跑通。这样后续接入记忆、限流、监控、真实 LLM 时，每一步都有稳定的验收点。

---

## 2. 当前目录应该怎么读

核心目录：

```text
05_PRODUCT_AGENT/
├── api/
│   ├── main.py          # FastAPI 入口：/、/health、/chat
│   ├── schemas.py       # /chat 请求和响应模型
│   ├── settings.py      # .env 配置读取
│   └── ui.py            # 内置客服工作台 HTML/CSS/JS
├── agent/
│   ├── graph.py         # LangGraph 工作流装配
│   ├── state.py         # CustomerServiceState
│   ├── nodes.py         # context_loader / agent / finalizer 节点
│   ├── intent.py        # 订单号和意图识别
│   ├── service.py       # 规则型客服决策
│   ├── tools.py         # Mock 业务工具
│   └── prompts.py       # 客服提示词占位
├── tests/
│   ├── test_chat_flow.py      # /chat 客服闭环测试
│   ├── test_tools.py          # Mock 工具测试
│   ├── test_ui.py             # UI 页面测试
│   ├── test_graph_skeleton.py # LangGraph 骨架测试
│   ├── test_health.py         # 健康检查测试
│   └── test_settings.py       # 配置默认值测试
├── README.md
└── DEV_PROGRESS.md
```

最重要的依赖方向：

```text
api -> agent.graph -> agent.nodes -> agent.service -> agent.intent / agent.tools
```

这个依赖方向说明：API 层不直接写客服业务规则，业务规则集中在 `agent/service.py`，业务数据访问集中在 `agent/tools.py`。

---

## 3. 一次请求的完整流程

以用户输入“我的订单 ORD123456 到哪了？”为例，当前完整链路是：

```text
浏览器 GET /
  -> api/ui.py 返回客服工作台
  -> 前端 fetch("/chat")
  -> api/main.py::chat()
  -> ChatRequest 校验输入
  -> customer_service_graph.invoke(...)
  -> agent/graph.py 进入 LangGraph
  -> context_loader_node 初始化上下文
  -> agent_node 读取最新 HumanMessage
  -> handle_customer_message 做意图判断
  -> get_logistics / get_order 读取 Mock 数据
  -> finalizer_node 计算窗口大小、轮次、响应耗时
  -> ChatResponse 返回 answer / order_context / handoff / quality_score
  -> UI 更新聊天区、订单上下文和运行状态
```

流程和代码对照：

| 流程 | 关键代码 | 解释 |
|---|---|---|
| 页面入口 | `api/main.py::customer_service_workspace` | 返回 `api/ui.py` 里的静态 HTML |
| 前端提交 | `api/ui.py::sendMessage` | 用 `fetch("/chat")` 调接口 |
| 请求校验 | `api/schemas.py::ChatRequest` | 校验 `user_id`、`session_id`、`message` |
| API 进入图 | `api/main.py::chat` | 把用户输入包装成 `HumanMessage` |
| 图编排 | `agent/graph.py::build_customer_service_graph` | 定义 `context_loader -> agent -> finalizer` |
| 状态初始化 | `agent/nodes.py::context_loader_node` | 补齐状态默认值、记录开始时间 |
| 客服决策 | `agent/nodes.py::agent_node` | 调 `handle_customer_message()` |
| 意图判断 | `agent/service.py::handle_customer_message` | 按优先级处理转人工、退款、物流、订单、商品 |
| 工具执行 | `agent/tools.py` | 返回 Mock 订单、物流、商品和退款结果 |
| 响应收尾 | `agent/nodes.py::finalizer_node` | 计算消息窗口、轮次、响应时间 |
| 响应模型 | `api/schemas.py::ChatResponse` | 固定 `/chat` 输出结构 |

面试重点：

> 这里的关键不是“写了几个 if”，而是把接口、状态机、业务决策、工具访问和 UI 展示拆成了不同层。这样后续把规则型决策替换成真实 LLM 或 ToolNode 时，API 契约和测试可以基本保持稳定。

---

## 4. API 层：FastAPI 如何接住请求

文件：`api/main.py`

关键代码：

```python
customer_service_graph = build_customer_service_graph()

app = FastAPI(title=settings.app_name, version=settings.app_version)
```

含义：

- 服务启动时构建一份 LangGraph。
- `/chat` 请求复用这份图，不在每次请求里重新组装节点。
- 当前图还没有持久化 checkpointer，所以 `session_id` 主要用于接口和后续 M2 扩展。

页面入口：

```python
@app.get("/", response_class=HTMLResponse)
async def customer_service_workspace() -> str:
    return CUSTOMER_SERVICE_UI
```

这让项目不需要单独启动 React、Vue 或 Streamlit。M1 阶段一个 FastAPI 进程即可同时提供 API 和演示 UI。

对话接口：

```python
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = customer_service_graph.invoke(
        {
            "session_id": req.session_id,
            "user_id": req.user_id,
            "messages": [HumanMessage(content=req.message)],
        },
        config={"configurable": {"thread_id": req.session_id}},
    )
```

解释：

- `ChatRequest` 负责输入校验。
- 用户文本被转成 LangChain 的 `HumanMessage`。
- `thread_id` 使用 `session_id`，这是为 M2 会话状态持久化预留的接口形状。
- 当前 `build_customer_service_graph()` 没有传入 checkpointer，所以它还不保存历史会话。

返回响应：

```python
return ChatResponse(
    session_id=result.get("session_id", req.session_id),
    user_id=result.get("user_id", req.user_id),
    answer=answer,
    needs_human_transfer=result.get("needs_human_transfer", False),
    transfer_reason=result.get("transfer_reason", ""),
    order_context=result.get("order_context"),
    token_used=result.get("token_used", 0),
    response_time_ms=result.get("response_time_ms", 0),
    quality_score=result.get("quality_score"),
)
```

面试重点：

> API 层只做协议转换：HTTP JSON -> LangGraph State -> HTTP JSON。它不直接实现订单查询、退款规则和转人工判断，这样边界更清晰。

---

## 5. Schema 层：为什么要单独定义请求和响应

文件：`api/schemas.py`

请求模型：

```python
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value
```

这里有两个层次的校验：

- `min_length=1`：防止字段完全为空。
- `value.strip()`：防止 `"   "` 这种空白输入。

测试对应：

```python
def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={"user_id": "user_001", "session_id": "session_001", "message": "   "},
    )
    assert response.status_code == 422
```

响应模型：

```python
class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    answer: str
    needs_human_transfer: bool
    transfer_reason: str
    order_context: dict[str, Any] | None
    token_used: int
    response_time_ms: int
    quality_score: int | None
```

面试重点：

> 生产系统里接口返回结构要稳定。即使当前 `quality_score` 和 `token_used` 还是占位，也先固定字段，后续接入真实评估和 Token 统计时不会破坏前端和调用方。

---

## 6. LangGraph 层：为什么 M1 也要用图

文件：`agent/graph.py`

当前图很简单：

```python
workflow = StateGraph(CustomerServiceState)

workflow.add_node("context_loader", context_loader_node)
workflow.add_node("agent", agent_node)
workflow.add_node("finalizer", finalizer_node)

workflow.add_edge(START, "context_loader")
workflow.add_edge("context_loader", "agent")
workflow.add_edge("agent", "finalizer")
workflow.add_edge("finalizer", END)
```

当前流程：

```text
START -> context_loader -> agent -> finalizer -> END
```

为什么不用普通函数直接写完？

- M1 虽然简单，但 M2 要加入记忆加载和裁剪。
- M3 要加入限流、预算、LLM fallback。
- M4 要加入质量评估和指标记录。
- LangGraph 提前把“客服系统是状态流转”这个架构固定下来。

面试重点：

> 这个阶段的图看起来简单，但它是为后续生产能力留扩展点。真实生产 Agent 往往不是一次函数调用，而是一条可观测、可插拔、可恢复的状态流。

---

## 7. State 层：CustomerServiceState 承载什么

文件：`agent/state.py`

关键代码：

```python
class CustomerServiceState(TypedDict, total=False):
    session_id: str
    user_id: str

    messages: Annotated[list[BaseMessage], add_messages]
    window_size: int
    total_turns: int

    user_profile: dict[str, Any]
    user_memories: list[str]
    order_context: dict[str, Any] | None

    needs_human_transfer: bool
    transfer_reason: str

    token_used: int
    response_time_ms: int
    quality_score: int | None
    tool_name: str
```

最关键的是这一行：

```python
messages: Annotated[list[BaseMessage], add_messages]
```

含义：

- `messages` 不是普通 list 覆盖。
- LangGraph 合并节点输出时会用 `add_messages` reducer，把新消息追加进去。
- 所以 `agent_node` 返回 `AIMessage` 后，最终 state 中会同时有用户消息和客服消息。

M1 中还没有真正的长期记忆，但字段已经预留：

- `user_profile`
- `user_memories`
- `order_context`

面试重点：

> LangGraph 的核心是 State。State 设计要提前考虑哪些字段会被多个节点读写、哪些字段要累积、哪些字段只是本轮中间数据。`messages` 使用 reducer 是为了让对话历史天然可追加。

---

## 8. Node 层：三个节点分别负责什么

文件：`agent/nodes.py`

### 8.1 context_loader_node

```python
def context_loader_node(state: CustomerServiceState) -> dict:
    messages = list(state.get("messages") or [])
    return {
        "session_id": state.get("session_id", ""),
        "user_id": state.get("user_id", ""),
        "window_size": len(messages),
        "total_turns": state.get("total_turns", _count_human_turns(messages)),
        "_started_at": time.perf_counter(),
    }
```

它的职责是：

- 补齐默认字段。
- 计算当前消息窗口大小。
- 统计用户轮次。
- 记录开始时间，给 finalizer 算响应耗时。

### 8.2 agent_node

```python
latest_human = next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)
if latest_human is not None:
    decision = handle_customer_message(str(latest_human.content))
    return {
        "messages": [AIMessage(content=decision.answer)],
        "order_context": decision.order_context,
        "needs_human_transfer": decision.needs_human_transfer,
        "transfer_reason": decision.transfer_reason,
        "quality_score": decision.quality_score,
        "tool_name": decision.tool_name,
        "token_used": max(1, len(str(latest_human.content)) // 2),
    }
```

它的职责是：

- 找到最新用户消息。
- 调用 `handle_customer_message()` 做客服决策。
- 把决策结果转换成 LangGraph state patch。
- 返回新的 `AIMessage`，由 `add_messages` 追加到消息列表。

### 8.3 finalizer_node

```python
def finalizer_node(state: CustomerServiceState) -> dict:
    messages = list(state.get("messages") or [])
    started_at = state.get("_started_at")
    response_time_ms = 0
    if isinstance(started_at, float):
        response_time_ms = max(0, int((time.perf_counter() - started_at) * 1000))

    return {
        "window_size": len(messages),
        "total_turns": _count_human_turns(messages),
        "response_time_ms": response_time_ms,
    }
```

它的职责是收尾，不做业务判断：

- 更新最终窗口大小。
- 更新总轮次。
- 记录响应耗时。

面试重点：

> 节点要单一职责。`context_loader` 负责上下文准备，`agent` 负责业务决策，`finalizer` 负责运行指标收尾。这样以后加记忆、监控或质量评估时，不需要把一个大函数拆开重构。

---

## 9. 意图识别层：为什么先做规则

文件：`agent/intent.py`

订单号识别：

```python
ORDER_ID_RE = re.compile(r"\bORD\d{6,}\b", re.IGNORECASE)

def extract_order_id(text: str) -> str:
    match = ORDER_ID_RE.search(text)
    return match.group(0).upper() if match else ""
```

关键词判断：

```python
def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(keyword.casefold() in normalized for keyword in keywords)
```

当前支持的意图：

- `is_human_transfer_request`
- `is_refund_request`
- `is_refund_confirmed`
- `is_logistics_query`
- `is_order_query`
- `is_product_query`

为什么 M1 不直接用 LLM 判断意图？

- 离线可测，不依赖密钥和网络。
- 行为可预测，适合建立测试基线。
- 客服关键规则，比如退款二次确认和转人工边界，必须确定性更强。
- 后续接真实 LLM 时可以用这些规则作为 guardrail。

面试重点：

> 不是所有 Agent 逻辑都应该交给 LLM。涉及退款、投诉、法律和转人工的规则，最好先有确定性安全边界，再让 LLM 做自然语言表达或复杂理解。

---

## 10. 客服决策层：M1 的业务核心

文件：`agent/service.py`

返回对象：

```python
@dataclass(frozen=True)
class CustomerServiceDecision:
    answer: str
    order_context: dict[str, Any] | None
    needs_human_transfer: bool
    transfer_reason: str
    quality_score: int
    tool_name: str
```

这个对象把客服决策标准化：

- `answer`：给用户看的回答。
- `order_context`：给 UI 或后续节点看的结构化上下文。
- `needs_human_transfer`：是否转人工。
- `transfer_reason`：为什么转人工。
- `quality_score`：M4 前的占位分数。
- `tool_name`：本轮用了什么工具或路径。

### 10.1 决策优先级

`handle_customer_message()` 的处理顺序是：

```text
转人工/投诉/法律
  -> 退款
  -> 物流
  -> 订单
  -> 商品
  -> fallback
```

为什么转人工优先？

- 用户明确要求人工时，系统不应该继续硬答。
- 投诉、法律、纠纷属于客服安全边界。
- 这也是 05 项目相比普通 Agent 更偏生产场景的体现。

### 10.2 退款必须二次确认

未确认时：

```python
refund = apply_refund(order_id, confirmed=is_refund_confirmed(message))
...
answer = (
    f"订单 {order_id} 当前可以发起退款。退款会进入人工复核，"
    "请回复“确认退款”后我再提交申请。"
)
```

确认后：

```python
if refund["refund_status"] == "submitted":
    answer = (
        f"订单 {order_id} 的退款申请已提交，工单号 {refund['refund_ticket_id']}。"
        "预计 1-3 个工作日内完成审核。"
    )
```

对应测试：

```python
def test_refund_requires_explicit_confirmation_before_submit():
    first = _chat("我要给订单 ORD123456 退款", session_id="refund_session")
    assert "确认" in first["answer"]
    assert "已提交" not in first["answer"]

    confirmed = _chat("我确认退款 ORD123456", session_id="refund_session")
    assert "退款申请已提交" in confirmed["answer"]
```

面试重点：

> 客服系统不能把“用户提到退款”直接理解成“用户确认提交退款”。M1 用显式确认保护了高风险动作，这是生产 Agent 中非常重要的安全边界。

### 10.3 物流和订单为什么分开

物流查询：

```python
if order_id and is_logistics_query(message):
    logistics = get_logistics(order_id)
    order = get_order(order_id)
    context = {**order, **logistics}
```

订单查询：

```python
if order_id and is_order_query(message):
    order = get_order(order_id)
```

区别：

- 订单查询回答商品、金额、订单状态、预计送达。
- 物流查询回答承运商、运单号、当前位置、物流事件。
- `order_context` 在物流场景会合并订单和物流信息，方便 UI 右侧展示。

面试重点：

> M1 虽然是 Mock 数据，但已经把“自然语言回答”和“结构化上下文”分开了。真实系统里结构化上下文可以继续给监控、转人工、后续记忆和质检使用。

---

## 11. 工具层：Mock 工具为什么也要像正式工具一样写

文件：`agent/tools.py`

当前有四类工具：

```python
def get_order(order_id: str) -> dict
def get_logistics(order_id: str) -> dict
def get_product(query: str) -> dict
def apply_refund(order_id: str, *, confirmed: bool) -> dict
```

### 11.1 为什么返回 deepcopy

```python
def get_order(order_id: str) -> dict:
    order = MOCK_ORDERS.get(order_id.upper())
    if order is None:
        return {"order_id": order_id.upper(), "status": "未找到"}
    return deepcopy(order)
```

原因：

- 防止调用方修改 Mock 数据源。
- 保持工具函数无副作用。
- 这和真实数据库读取后的 DTO 思路接近。

### 11.2 退款工具的 confirmed 参数

```python
def apply_refund(order_id: str, *, confirmed: bool) -> dict:
    ...
    if not confirmed:
        return {
            **order,
            "refund_status": "confirmation_required",
            "message": "退款会进入人工复核，请确认是否继续提交退款申请。",
        }
```

`confirmed` 用关键字参数强制传入，避免误调用：

```python
apply_refund("ORD123456", confirmed=True)
```

比下面这种更安全：

```python
apply_refund("ORD123456", True)
```

面试重点：

> Mock 工具不是随便返回字符串。它应该尽量模拟真实业务工具的接口：输入参数明确、输出结构化、失败路径可预期、危险动作需要显式确认。

---

## 12. UI 层：为什么用 FastAPI 内置页面

文件：`api/ui.py`

当前 UI 是一个单文件 HTML：

- 左侧：用户 ID、会话 ID、快捷问题。
- 中间：聊天区和输入框。
- 右侧：质量分、Token、转人工状态、转人工原因、订单上下文。

核心前端调用：

```javascript
const response = await fetch("/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    user_id: document.querySelector("#userId").value,
    session_id: document.querySelector("#sessionId").value,
    message,
  }),
});
```

响应后更新状态：

```javascript
document.querySelector("#quality").textContent = payload.quality_score ?? "-";
document.querySelector("#tokens").textContent = payload.token_used ?? "-";
document.querySelector("#handoff").textContent = payload.needs_human_transfer ? "是" : "否";
document.querySelector("#context").textContent = JSON.stringify(payload.order_context || {}, null, 2);
```

为什么 M1 不用 React/Vue？

- 当前目标是客服闭环，不是复杂前端工程。
- 一个 FastAPI 服务即可演示，启动成本低。
- UI 只需要验证核心交互：发消息、看回答、看上下文、看转人工状态。

面试重点：

> M1 的 UI 是为“演示和调试 Agent 行为”服务的，不是产品级前端。它把 `order_context`、`quality_score`、`token_used` 这些后端字段可视化，方便说明生产 Agent 不只是聊天，还要暴露可运营信息。

---

## 13. 测试层：每个测试在保护什么

当前测试共 17 个，核心测试文件如下。

### 13.1 `tests/test_chat_flow.py`

覆盖客服主流程：

- 订单状态：`test_chat_answers_order_status`
- 物流状态：`test_chat_answers_logistics_status`
- 商品库存：`test_chat_answers_product_question`
- 退款确认：`test_refund_requires_explicit_confirmation_before_submit`
- 转人工：`test_chat_marks_human_transfer_for_complaint_and_legal_issue`
- 中英混合：`test_chat_handles_mixed_english_and_chinese`
- 空消息校验：`test_chat_rejects_empty_message`

这组测试说明 M1 的验收标准不是口头描述，而是可重复运行的自动化检查。

### 13.2 `tests/test_tools.py`

覆盖 Mock 工具：

- `get_order`
- `get_logistics`
- `get_product`
- `apply_refund`
- `list_customer_service_tools`

面试时可以强调：

> 工具层单独测试，能避免所有问题都要通过 `/chat` 黑盒排查。业务工具错了，就先在工具测试里定位。

### 13.3 `tests/test_ui.py`

```python
def test_root_serves_customer_service_workspace():
    response = client.get("/")
    assert response.status_code == 200
    assert "智能客服工作台" in response.text
    assert "/chat" in response.text
    assert "订单上下文" in response.text
```

这个测试不验证浏览器交互细节，但保护了 M1 的基本 UI 入口不被破坏。

### 13.4 `tests/test_graph_skeleton.py`

保护 LangGraph 能编译、节点存在、离线调用能返回客服消息。

面试重点：

> 测试按层分布：API 流程测试、工具单元测试、UI 入口测试、图结构测试。这样比只做端到端测试更容易定位问题。

---

## 14. 当前实现和设计稿的差异

这一节面试时很重要，因为它能体现你对项目状态诚实且清楚。

### 14.1 当前没有真实 LLM

当前是规则型客服决策：

```text
agent_node -> handle_customer_message -> intent/tools
```

不是：

```text
agent_node -> LLM tool calling -> ToolNode -> LLM final answer
```

原因是 M1 目标是稳定闭环和测试基线，真实 LLM 放到后续迭代更合适。

### 14.2 当前没有持久化会话

虽然 `/chat` 传了：

```python
config={"configurable": {"thread_id": req.session_id}}
```

但 `build_customer_service_graph()` 当前没有传入 SQLite checkpointer，所以服务重启或多轮状态还不会恢复。

这正是 M2 要做的内容。

### 14.3 当前工具不是 LangChain ToolNode

当前工具是普通 Python 函数，由 `agent/service.py` 直接调用。

好处：

- 简单可测。
- 不依赖 LLM tool calling。
- 适合建立业务规则基线。

后续可以演进为：

```text
agent -> tools -> agent -> finalizer
```

或者保留规则 guardrail，让 LLM 只处理表达和复杂语义。

### 14.4 质量分和 Token 是占位

当前：

- `quality_score` 是规则路径给出的固定分值。
- `token_used` 是基于输入长度的轻量估算。

M4 才会接入真实质量评估和指标系统。

面试讲法：

> 我会明确区分已实现和计划实现。当前 M1 的价值是把接口、图、业务规则、工具和 UI 闭环打通；生产级能力会在 M2-M5 逐层补齐。

---

## 15. 面试高频问题与回答

### Q1：为什么这个项目叫生产级 Agent，但现在不用真实 LLM？

回答：

> 生产级不等于第一步就调用真实模型。生产级更重要的是接口稳定、行为可测、安全边界清楚、可观测字段预留和迭代路径明确。M1 先用离线规则跑通客服闭环，保证退款确认、转人工、订单上下文这些关键行为稳定，再逐步接入真实 LLM、记忆、限流和监控。

### Q2：为什么要用 LangGraph，而不是 FastAPI 里一个函数写完？

回答：

> 因为客服 Agent 后续会变成多节点状态流：记忆加载、上下文裁剪、LLM 推理、工具调用、质量评估、记忆保存、转人工。M1 的图虽然只有三步，但已经把状态结构和节点边界固定了，后续扩展不用重写 API 契约。

### Q3：退款为什么要二次确认？

回答：

> 退款属于有业务影响的动作，不能因为用户提到“退款”就直接提交。当前实现要求用户明确确认后才调用 `apply_refund(..., confirmed=True)`，这是高风险动作的安全边界。真实生产里还会结合权限、订单状态、风控和人工复核。

### Q4：转人工规则为什么优先级最高？

回答：

> 用户明确要求人工、投诉、法律、纠纷等场景继续让 AI 硬答会有业务风险。当前 `handle_customer_message()` 先判断转人工，再判断退款、物流、订单和商品，体现的是客服系统的安全优先级。

### Q5：Mock 工具有实际意义吗？

回答：

> 有。Mock 工具让 Agent 闭环可以离线测试，同时约束了未来真实工具的接口形状。比如 `get_order()`、`get_logistics()`、`apply_refund()` 都返回结构化 dict，后续替换成数据库或外部 API 时，service 层和 API 层可以少改。

### Q6：当前 `/chat` 的返回字段为什么包含 `quality_score` 和 `token_used`？

回答：

> 这些是为生产运营预留的字段。M1 是占位值，M4 会接真实评估和 Prometheus 指标。提前固定字段可以让 UI、测试和调用方先适配生产系统需要关注的指标。

### Q7：当前项目的最大短板是什么？

回答：

> 最大短板是还没有 M2-M4 的生产能力：会话状态不持久、没有长期记忆、没有 Redis 限流、没有真实 LLM fallback、没有 Prometheus 和 LangSmith。当前 M1 只能说明客服闭环和工程骨架成立，不能宣称已经能承接真实流量。

### Q8：如果继续开发 M2，你会怎么做？

回答：

> 我会先接短期记忆和 SQLite checkpointer，让 `session_id` 真正能恢复会话；再做消息窗口裁剪和摘要；最后用稳定接口封装长期记忆，先支持保存、检索和删除用户记忆。测试会重点覆盖多轮对话、服务重启恢复和删除后不可召回。

---

## 16. 面试时可以画的架构图

```text
Browser UI
   |
   | GET /
   v
FastAPI UI Page
   |
   | fetch POST /chat
   v
FastAPI API Layer
   |
   | ChatRequest -> HumanMessage
   v
LangGraph StateGraph
   |
   | context_loader -> agent -> finalizer
   v
Rule-based Customer Service Decision
   |
   | intent + mock tools
   v
Mock Business Tools
   |
   | order / logistics / product / refund
   v
ChatResponse
   |
   | answer + context + handoff + metrics
   v
Browser UI
```

这张图的讲解重点：

- UI 和 API 在同一个 FastAPI 服务中。
- API 不直接做业务判断，而是进入 LangGraph。
- LangGraph 当前是线性图，但为后续记忆、工具节点、质量评估预留了节点扩展。
- 业务工具返回结构化数据，不只是拼接字符串。
- 响应包含客服答案和运营字段。

---

## 17. 你应该能现场演示什么

启动：

```bash
cd 05_PRODUCT_AGENT
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

推荐演示顺序：

1. 点击“订单状态”或输入：`我的订单 ORD123456 到哪了？`
2. 点击“物流查询”，观察右侧订单上下文。
3. 点击“商品库存”，观察 `order_context` 为空但回答正常。
4. 点击“退款申请”，说明不会直接提交。
5. 输入：`我确认退款 ORD123456`，说明确认后才提交。
6. 点击“转人工”，观察右侧转人工状态和原因。

接口演示：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","session_id":"session_001","message":"Where is my order ORD123456？谢谢"}'
```

测试演示：

```bash
pytest tests -q
```

当前应通过 17 个测试。

---

## 18. 常见误区

### 误区 1：把当前项目说成已经生产可用

不准确。当前只是 M1。真实生产还缺：

- 会话持久化
- 长期记忆
- 限流和 Token 预算
- LLM fallback
- 监控指标
- Docker Compose 和压测

### 误区 2：把规则型决策说成 LLM Agent

当前不是 LLM 推理型 Agent，而是 LangGraph 编排下的规则型客服 MVP。更准确的说法是：

> 当前 M1 用规则型决策模拟客服 Agent 的业务闭环，后续会把 agent 节点替换或扩展为真实 LLM + tool calling。

### 误区 3：忽略测试

这个项目当前最有价值的部分之一就是测试已经锁住 M1 验收标准。面试时不要只讲 UI 和接口，也要讲：

- 空消息 422
- 退款确认
- 转人工原因
- 中英文混合输入
- 工具函数返回结构化数据

### 误区 4：忽略 `order_context`

`order_context` 是连接“自然语言客服回答”和“结构化业务系统”的关键。后续转人工、质检、记忆保存都可以复用它。

---

## 19. 复盘总结

当前 05 项目的已开发部分可以总结为三句话：

1. M0 建好了可运行、可测试、可扩展的 FastAPI + LangGraph 骨架。
2. M1 跑通了客服核心闭环：订单、物流、商品、退款确认、转人工和 UI 演示。
3. 当前实现保持离线可测，把生产级能力拆成后续迭代逐步加入，而不是一开始堆满不可验证的基础设施。

面试收尾表达：

> 我在这个项目里关注的不是“让模型回答一句话”，而是把客服 Agent 做成一个可以持续演进的服务：有稳定 API、有显式状态、有工具边界、有安全规则、有 UI 演示、有测试基线。M1 先验证闭环，后续 M2-M5 再逐层加入记忆、限流、弹性、观测和部署能力。
