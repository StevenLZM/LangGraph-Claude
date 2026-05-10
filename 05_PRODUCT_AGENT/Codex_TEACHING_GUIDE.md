# 05 Product Agent 教学文档

这份文档用于讲解 `05_PRODUCT_AGENT` 当前已经完成的 M0-M6。重点不是背目录，而是能把一次客服请求从 UI 到 API、记忆、限流、LangGraph、规则决策、DeepSeek 生成、FAQ/RAG、质量评估、指标记录、Docker Compose 部署、Locust 压测和自动评测完整讲清楚。

当前真实完成范围：

- M0：项目骨架、FastAPI 健康检查、基础配置、最小 LangGraph 图、pytest 测试。
- M1：`POST /chat`、内置客服 UI、Mock 订单/物流/商品/退款工具、规则型客服 Agent、转人工标记、退款二次确认。
- M2：短期记忆窗口、会话状态、用户长期记忆、`GET /sessions/{session_id}`、`DELETE /users/{user_id}/memories`；当前本地默认 SQLite，Docker Compose 默认 Postgres。
- M3：Hybrid 限流、全局 QPS 控制、Token 预算降级、`ResilientLLM` 重试/备用模型/熔断测试层。
- M4：`GET /metrics`、Prometheus 兼容指标、LangSmith trace metadata、自动质量评估、低质量告警事件。
- M5：Dockerfile、Docker Compose、Prometheus/Grafana 编排、Grafana 看板 provisioning、Locust 压测入口。
- M6：DeepSeek 真实 LLM 主路径、FAQ/RAG 适配、管理接口、100 题评测集、自动评测报告。

尚未完成范围：

- 24 小时长稳运行、正式 Docker 压测报告、pgvector/Mem0 语义记忆迁移和外部告警通道。

面试时要明确：当前 M6 已具备本地可演示的“客服闭环 + 记忆 + 限流 + 观测 + 质量评估 + DeepSeek + FAQ/RAG + 管理接口 + 容器编排 + Postgres 业务存储 + LangGraph checkpointer + 压测入口 + 自动评测”能力，但还不是完整线上生产系统；正式压测报告、24 小时长稳验证、pgvector 语义记忆迁移和外部告警通道仍需后续环境验证。

---

## 1. 项目一句话介绍

`05_PRODUCT_AGENT` 是一个生产级 AI 客服系统的渐进式实现。当前阶段已经跑通可测试、可部署、可评测的客服服务：用户从 UI 或 API 发起咨询，FastAPI 做协议校验、限流和 Token 预算，加载会话与用户记忆，进入带可选 checkpointer 的 LangGraph 状态流，规则型客服决策调用 Mock 业务工具或 FAQ/RAG 适配层，然后必须通过 DeepSeek/真实 LLM 基于工具结果生成最终话术，最后保存会话、评估回答质量、记录 Prometheus 兼容指标并返回响应。M5 提供 Docker Compose、Prometheus、Grafana 和 Locust 压测入口，M6 补齐 DeepSeek、管理接口和 100 题自动评测，存储升级补齐 Postgres 业务存储和 LangGraph Postgres/Redis checkpointer。

面试讲法：

> 这个项目不是一开始就追求真实 LLM 和全套 K8s 部署，而是按生产能力拆迭代：先稳定客服闭环和接口契约，再加记忆、限流、预算、弹性、观测和质量评估。这样每个阶段都有自动化测试和可演示验收点。

---

## 2. 当前目录应该怎么读

核心目录：

```text
05_PRODUCT_AGENT/
├── api/
│   ├── main.py                  # FastAPI 入口：/、/health、/chat、/metrics
│   ├── schemas.py               # /chat 请求和响应模型
│   ├── settings.py              # .env 配置读取
│   ├── ui.py                    # 内置客服工作台 HTML/CSS/JS
│   ├── routers/
│   │   └── admin.py             # M6 管理接口
│   └── middleware/
│       └── rate_limiter.py      # 用户限流、QPS、Token 预算
├── agent/
│   ├── checkpointing.py         # LangGraph Postgres/Redis/none checkpointer 工厂
│   ├── graph.py                 # LangGraph 工作流装配
│   ├── state.py                 # CustomerServiceState
│   ├── nodes.py                 # context_loader / agent / finalizer 节点
│   ├── intent.py                # 订单号和意图识别
│   ├── service.py               # 规则型客服决策
│   └── tools.py                 # Mock 业务工具
├── memory/
│   ├── short_term.py            # 摘要 + 最近 8 轮的短期窗口
│   ├── factory.py               # SQLite/Postgres 存储后端选择
│   ├── session_store.py         # SQLite/Postgres 会话状态
│   └── long_term.py             # SQLite/Postgres 用户长期记忆
├── llm/
│   ├── factory.py               # DeepSeek/OpenAI-compatible/Anthropic 客户端工厂
│   └── resilient_llm.py         # 主备模型、重试、熔断
├── rag/
│   └── faq_tool.py              # M6 FAQ/RAG 适配 01_RAG
├── evals/
│   ├── dataset.jsonl            # M6 100 题评测集
│   ├── run.py                   # 自动评测执行
│   └── report.py                # Markdown 报告生成
├── monitoring/
│   ├── metrics.py               # Prometheus 兼容指标
│   ├── tracing.py               # LangSmith trace metadata
│   └── evaluator.py             # 自动质量评估和低质告警
├── infra/
│   ├── prometheus.yml           # Prometheus scrape 配置
│   └── grafana/                 # Grafana datasource/dashboard provisioning
├── load_tests/
│   └── locustfile.py            # Locust 压测场景
├── Dockerfile
├── docker-compose.yml
├── tests/
│   ├── test_chat_flow.py
│   ├── test_memory.py
│   ├── test_rate_limiter.py
│   ├── test_llm_fallback.py
│   ├── test_observability.py
│   └── test_deployment.py
├── README.md
└── DEV_PROGRESS.md
```

最重要的依赖方向：

```text
api.main
  -> agent.checkpointing
  -> api.routers.admin
  -> api.middleware.rate_limiter
  -> memory.factory -> memory.session_store / memory.long_term / memory.short_term
  -> agent.graph -> agent.nodes -> agent.service -> agent.intent / agent.tools / rag.faq_tool
  -> llm.factory / llm.resilient_llm
  -> monitoring.tracing / monitoring.evaluator / monitoring.metrics
```

这个依赖方向说明：API 层是生产边界，负责协议、限流、记忆、观测和响应落库；客服业务规则集中在 `agent/service.py`；业务数据访问集中在 `agent/tools.py`；运营指标集中在 `monitoring/`。

---

## 3. 一次 `/chat` 请求的完整流程

以用户输入“我的订单 ORD123456 到哪了？”为例，当前完整链路是：

```text
浏览器 GET /
  -> api/ui.py 返回客服工作台
  -> 前端 fetch("/chat")
  -> api/main.py::chat()
  -> ChatRequest 校验输入
  -> RateLimiter 检查用户限流和全局 QPS
  -> SessionStore 加载历史会话
  -> UserMemoryManager 召回用户长期记忆
  -> ContextWindowManager 裁剪短期上下文并估算 token
  -> RateLimiter 预留单次和全局 token 预算
  -> build_trace_config 生成 session/user metadata
  -> customer_service_graph.invoke(..., configurable.thread_id=session_id)
  -> context_loader_node 初始化上下文
  -> agent_node 调 handle_customer_message 做客服决策
  -> get_order / get_logistics / get_product / apply_refund 读取 Mock 业务数据
  -> finalizer_node 计算窗口大小、轮次、响应耗时
  -> AutoQualityEvaluator 评估准确性、礼貌性、完整性
  -> UserMemoryManager 提取并保存长期记忆
  -> SessionStore 保存本轮消息和质量评估 metadata
  -> record_chat_request 记录 Prometheus 兼容指标
  -> ChatResponse 返回 answer / context / handoff / memories / quality_score / degrade 状态
  -> UI 更新聊天区、用户记忆、摘要、订单上下文和运行状态
```

流程和代码对照：

| 流程 | 关键代码 | 解释 |
|---|---|---|
| 页面入口 | `api/main.py::customer_service_workspace` | 返回 `api/ui.py` 里的静态 HTML |
| 请求校验 | `api/schemas.py::ChatRequest` | 校验 `user_id`、`session_id`、`message` |
| 限流/QPS | `api/middleware/rate_limiter.py::RateLimiter` | 用户每分钟限制和全局每秒限制 |
| 会话加载 | `memory/session_store.py::load_session` | 读取同一 `session_id` 的历史消息 |
| 记忆召回 | `memory/long_term.py::load_memories` | 按用户和当前问题召回长期记忆 |
| 窗口裁剪 | `memory/short_term.py::trim` | 摘要早期消息，保留最近 8 轮 |
| Token 预算 | `RateLimiter.reserve_token_budget` | 单次/全局预算超限走降级回复 |
| Trace metadata | `monitoring/tracing.py::build_trace_config` | 给 LangGraph 调用注入 session/user metadata |
| Checkpoint | `agent/checkpointing.py::build_checkpointer` | 按配置接入 none/Postgres/Redis checkpointer |
| 图编排 | `agent/graph.py::build_customer_service_graph` | 定义 `context_loader -> agent -> finalizer`，可注入 checkpointer |
| 客服决策 | `agent/service.py::handle_customer_message` | 按优先级处理转人工、退款、物流、订单、商品 |
| 工具执行 | `agent/tools.py` | 返回 Mock 订单、物流、商品和退款结果 |
| 质量评估 | `monitoring/evaluator.py::AutoQualityEvaluator` | 计算 `quality_score` 和低质告警 |
| 指标记录 | `monitoring/metrics.py::record_chat_request` | 记录请求数、延迟、Token、质量分等 |
| 响应模型 | `api/schemas.py::ChatResponse` | 固定 `/chat` 输出结构 |

面试重点：

> 这里的关键不是“写了几个 if”，而是把接口、限流、记忆、状态机、业务决策、工具访问、质量评估和指标记录拆成了不同层。当前已能在 SQLite/Postgres 间切换业务存储，并可注入 LangGraph checkpointer；后续把关键词记忆升级成 Mem0/pgvector，或者把规则型决策演进为 ToolNode 时，API 契约和测试可以保持稳定。

---

## 4. API 层：FastAPI 如何承担生产边界

文件：`api/main.py`

启动时构建核心单例：

```python
checkpointer = build_checkpointer(settings)
customer_service_graph = build_customer_service_graph(checkpointer=checkpointer)
context_window_manager = ContextWindowManager()
session_store = build_session_store(settings)
user_memory_manager = build_user_memory_manager(settings)
quality_evaluator = AutoQualityEvaluator(alert_threshold=settings.quality_alert_threshold)
rate_limiter = RateLimiter(...)
```

含义：

- 图、checkpointer、会话存储、长期记忆、限流器和质量评估器在进程内复用。
- `/chat` 不在每次请求重新构建图或初始化存储。
- 本地默认使用 SQLite、无 checkpointer 和内存限流，便于离线演示和测试；Docker Compose 默认使用 Postgres 业务存储、Postgres checkpointer 和 Redis 限流。

核心接口：

```python
@app.get("/")
async def customer_service_workspace() -> str

@app.get("/health")
async def health() -> dict

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse
```

`/chat` 的职责不是直接写业务逻辑，而是组织生产链路：

```text
校验 -> 限流 -> 记忆/会话 -> token 预算 -> 图调用 -> 质量评估 -> 指标 -> 保存 -> 响应
```

面试重点：

> API 层是生产边界。它接住 HTTP 请求，处理限流、预算、观测和持久化，再把业务判断交给 Agent 图和 service 层。这样服务治理能力不会散落在业务规则里。

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
    user_memories: list[str] = Field(default_factory=list)
    memory_summary: str = ""
    degraded: bool = False
    degrade_reason: str = ""
```

重点字段：

- `quality_score`：M4 后由 `AutoQualityEvaluator` 计算，不再是纯占位。
- `token_used`：当前仍是轻量估算；真实模型接入后应优先使用 provider usage。
- `degraded` / `degrade_reason`：M3 Token 预算超限时的显式降级信号。

面试重点：

> 生产系统里接口返回结构要稳定。`quality_score`、`token_used`、`degraded` 这些字段让前端、监控和调用方提前适配运营视角，而不是只拿一段自然语言回答。

---

## 6. LangGraph 层：为什么现在仍保持线性图

文件：`agent/graph.py`

当前图：

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

- M2 已经加入记忆加载、窗口裁剪和会话持久化。
- M3 在 API 边界加入限流、预算和降级，图内部保持业务状态流简单。
- M4 在 API 边界加入 trace metadata、质量评估和指标记录，图调用天然可被追踪。
- 后续如果引入 ToolNode、LLM agent、多轮工具调用或 quality node，图结构可以平滑扩展。

面试重点：

> 这个阶段的图看起来简单，但它固定了“客服系统是一条状态流”这个架构。生产能力不一定都塞进图里：限流和 HTTP 指标更适合 API 边界，业务推理和工具链路更适合图里。

---

## 7. State 层：CustomerServiceState 承载什么

文件：`agent/state.py`

关键结构：

```python
class CustomerServiceState(TypedDict, total=False):
    session_id: str
    user_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    window_size: int
    total_turns: int
    user_profile: dict[str, Any]
    user_memories: list[str]
    memory_summary: str
    order_context: dict[str, Any] | None
    needs_human_transfer: bool
    transfer_reason: str
    token_used: int
    response_time_ms: int
    quality_score: int | None
    tool_name: str
    _started_at: float
```

最关键的是：

```python
messages: Annotated[list[BaseMessage], add_messages]
```

含义：

- `messages` 不是普通 list 覆盖。
- LangGraph 合并节点输出时会用 `add_messages` reducer，把新消息追加进去。
- `agent_node` 返回 `AIMessage` 后，最终 state 中同时包含用户消息和客服消息。

面试重点：

> LangGraph 的核心是 State。State 设计要提前考虑哪些字段会被多个节点读写、哪些字段要累积、哪些字段只是本轮中间数据。`messages` 使用 reducer 是为了让对话历史天然可追加。

---

## 8. Node 层：三个节点分别负责什么

文件：`agent/nodes.py`

### 8.1 `context_loader_node`

职责：

- 补齐 `session_id`、`user_id`、记忆、转人工、token、质量分等默认字段。
- 计算当前消息窗口大小和用户轮次。
- 记录 `_started_at`，供 `finalizer_node` 计算响应耗时。

### 8.2 `agent_node`

职责：

- 找到最新用户消息。
- 调用 `handle_customer_message()` 做客服决策。
- 把决策结果转换成 LangGraph state patch。
- 返回新的 `AIMessage`，由 `add_messages` 追加到消息列表。
- 根据输入长度给出轻量 `token_used` 估算。

### 8.3 `finalizer_node`

职责：

- 更新最终窗口大小。
- 更新总轮次。
- 记录响应耗时。
- 不做业务判断。

面试重点：

> 节点要单一职责。`context_loader` 负责上下文准备，`agent` 负责业务决策，`finalizer` 负责运行指标收尾。质量评估和 Prometheus 指标目前放在 API 边界，是因为它们关注 HTTP 请求和最终响应，而不是单个图节点。

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

- 转人工/投诉/法律问题。
- 退款申请和退款确认。
- 物流查询。
- 订单查询。
- 商品查询。
- 记忆召回和偏好保存。

为什么当前主路径不直接用 LLM 判断意图？

- 离线可测，不依赖密钥和网络。
- 行为可预测，适合建立测试基线。
- 退款二次确认、投诉、法律和转人工是客服安全边界，确定性规则更可靠。
- 后续接真实 LLM 时可以用这些规则作为 guardrail。

面试重点：

> 不是所有 Agent 逻辑都应该交给 LLM。涉及退款、投诉、法律和转人工的规则，最好先有确定性安全边界，再让 LLM 做自然语言表达或复杂理解。

---

## 10. 客服决策层：业务核心

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
- `order_context`：给 UI、质量评估或后续节点看的结构化上下文。
- `needs_human_transfer`：是否转人工。
- `transfer_reason`：为什么转人工。
- `quality_score`：规则路径的初始分，M4 最终返回值由评估器计算。
- `tool_name`：本轮用了什么工具或路径。

决策优先级：

```text
转人工/投诉/法律
  -> 记忆召回
  -> 退款
  -> 物流
  -> 订单
  -> 商品
  -> 偏好保存
  -> fallback
```

### 10.1 退款必须二次确认

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

面试重点：

> 客服系统不能把“用户提到退款”直接理解成“用户确认提交退款”。当前实现要求用户明确确认后才调用 `apply_refund(..., confirmed=True)`，这是高风险动作的安全边界。

### 10.2 物流和订单为什么分开

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
- `order_context` 在物流场景会合并订单和物流信息，方便 UI、转人工和质检复用。

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

Mock 工具的设计要求：

- 输入参数明确。
- 输出结构化 dict。
- 找不到数据时返回可预期状态，不抛不可控异常。
- 退款这类危险动作必须显式传入 `confirmed=True`。
- 返回数据使用 copy，避免调用方污染 Mock 数据源。

面试重点：

> Mock 工具不是随便返回字符串。它应该尽量模拟真实业务工具的接口：输入参数明确、输出结构化、失败路径可预期、危险动作需要显式确认。这样后续替换成数据库或外部 API 时，上层改动最小。

---

## 12. M2 记忆系统：短期、会话、长期三层

M2 的核心变化是把“每次请求都像第一次见用户”升级为“能管理当前会话，也能跨会话记住用户关键信息”。

### 12.1 短期记忆：`ContextWindowManager`

文件：`memory/short_term.py`

核心策略：

```python
recent_messages = list(messages[-self.max_messages :])
old_messages = list(messages[: -self.max_messages])
summary = self._summarize(old_messages)
trimmed = [SystemMessage(content=summary)] + recent_messages
```

当前默认保留最近 16 条消息，也就是最近 8 轮用户/客服对话。更早的消息会被压缩成一条 `SystemMessage`：

```text
[早期对话摘要] 已压缩 N 条早期消息...
```

为什么这样做：

- 避免 100 轮对话把上下文无限撑大。
- 保留最近对话的细节。
- 用摘要保留早期对话的关键背景。
- 后续接真实 LLM 时，可以把摘要作为系统上下文。

### 12.2 会话状态：`SessionStore`

文件：`memory/session_store.py`

会话状态解决的是同一个 `session_id` 的连续对话和状态查询：

```python
session_store.save_session(
    session_id=req.session_id,
    user_id=req.user_id,
    messages=result.get("messages", []),
    metadata={
        "summary": memory_summary,
        "needs_human_transfer": result.get("needs_human_transfer", False),
        "transfer_reason": result.get("transfer_reason", ""),
        "token_used": token_used,
        "quality_score": quality_score,
        "quality_evaluation": evaluation.to_dict(),
        "quality_alert": not evaluation.passed,
    },
)
```

会话查询：

```python
@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    session = session_store.get_public_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session
```

当前会话状态分两层：

- 业务会话表：`SessionStore`/`PostgresSessionStore` 保存消息窗口、质量评估、转人工原因和 trace metadata，支撑 `/sessions` 与管理接口。
- LangGraph checkpoint：`agent/checkpointing.py` 按 `CHECKPOINTER_BACKEND` 接入 `none`、Postgres 或 Redis，用于图执行状态的原生 checkpoint/resume。

本地默认 `STORAGE_BACKEND=sqlite`、`CHECKPOINTER_BACKEND=none`，便于 pytest 和离线演示；Docker Compose 默认 `STORAGE_BACKEND=postgres`、`CHECKPOINTER_BACKEND=postgres`，更接近多实例部署。

### 12.3 长期记忆：`UserMemoryManager`

文件：`memory/long_term.py`

当前长期记忆保存三类信息：

- 偏好：例如“我喜欢顺丰配送”。
- 投诉：例如“我对物流很不满”。
- 用户资料：例如“我叫张三”。

入口：

```python
user_memories = user_memory_manager.load_memories(req.user_id, req.message)
user_memory_manager.save_from_turn(req.user_id, req.message, answer)
deleted = user_memory_manager.delete_memories(user_id)
```

面试重点：

> 短期记忆解决上下文窗口，会话状态解决同一 `session_id` 的连续性，长期记忆解决跨会话用户画像。长期记忆必须有删除接口，否则不满足用户可控性和隐私合规。

---

## 13. M3 限流与弹性：成本和失败影响控制

M3 的目标是控制成本和失败影响，让系统在高频请求、Token 超支、LLM 异常时仍能给出可控响应。

### 13.1 `RateLimiter` 的三类保护

文件：`api/middleware/rate_limiter.py`

当前能力：

- 用户级限流：默认同一用户每分钟 10 次请求，第 11 次返回 `429`。
- 全局 QPS：默认每秒 100 次请求，超限返回 `503`。
- Token 预算：单次默认 4000 tokens，全局每小时默认 500000 tokens。

Hybrid 策略：

```text
REDIS_URL 为空
  -> 使用进程内存计数，适合本地演示和测试

REDIS_URL 已配置
  -> 使用 Redis async counter，适合多进程/多实例部署
```

为什么要有内存后端？

- 本地开发不需要先启动 Redis。
- pytest 不依赖外部服务。
- 接口和行为先稳定，M5 已用 Docker Compose 补齐 Redis 服务编排。

### 13.2 Token 预算超限为什么返回降级 200

在 `/chat` 中先估算上下文 token，再预留预算：

```python
estimated_tokens = context_window_manager.count_tokens(messages)
try:
    await rate_limiter.reserve_token_budget(estimated_tokens)
except TokenBudgetExceeded as exc:
    return _build_degraded_chat_response(...)
```

单次或全局预算超限时，系统不崩溃，也不返回 500，而是返回简化客服回复：

```json
{
  "degraded": true,
  "degrade_reason": "single_request_token_budget_exceeded"
}
```

面试重点：

> 预算超限是生产系统的正常风险，不应该变成服务异常。M3 把它设计成可观测、可解释的降级响应：用户得到明确提示，调用方也能通过 `degraded` 字段知道本轮不是完整处理。

### 13.3 `ResilientLLM` 是测试层还是主路径

文件：`llm/resilient_llm.py`

当前实现了：

- 主模型失败后指数退避重试。
- 主模型持续失败后切换备用模型。
- 连续失败达到阈值后熔断，短时间内优先走备用模型。
- 可注入 fake client，便于离线测试 fallback、retry、circuit breaker。

当前客服主路径仍是规则型离线实现，`ResilientLLM` 是可测试的弹性层，不直接影响 `/chat` 的客服回答。

面试重点：

> M3 先把模型调用的弹性边界做成可测试组件。等主路径切到真实 LLM 时，Agent 节点不应该直接调用裸模型，而应该通过 `ResilientLLM` 这一层处理重试、备用模型和熔断。

---

## 14. M4 可观测性与质量评估

M4 的目标是让系统可运营：每次对话可追踪，关键指标可监控，低质量回答可发现。

### 14.1 LangSmith trace metadata

文件：`monitoring/tracing.py`

核心函数：

```python
def build_trace_config(
    *,
    session_id: str,
    user_id: str,
    environment: str,
    app_version: str,
) -> dict[str, Any]:
    return {
        "tags": ["customer-service", f"session:{session_id}", f"user:{user_id}"],
        "metadata": {
            "session_id": session_id,
            "user_id": user_id,
            "environment": environment,
            "version": app_version,
        },
    }
```

在图调用中注入：

```python
result = customer_service_graph.invoke(
    state,
    config={
        "configurable": {"thread_id": req.session_id},
        "tags": trace_config["tags"],
        "metadata": trace_config["metadata"],
    },
)
```

本地默认 `LANGCHAIN_TRACING_V2=false`，不会强制访问外部 LangSmith；配置密钥后可打开 tracing。

### 14.2 Prometheus 兼容指标

文件：`monitoring/metrics.py`

当前暴露：

- `agent_requests_total{status=...}`：请求数。
- `agent_response_time_seconds_count/sum`：响应时间。
- `agent_tokens_total{type="estimated"}`：估算 Token 消耗。
- `agent_active_sessions`：活跃会话数。
- `agent_quality_score_count/sum`：质量分。
- `agent_errors_total`：错误数。
- `agent_human_transfers_total`：转人工数。

接口：

```python
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return render_prometheus_metrics()
```

本地单进程运行时不强制依赖 Prometheus Server 或 Grafana。M5 已提供 Compose 编排，Prometheus 会抓取 `api:8000/metrics`，Grafana 会自动加载客服运营看板。

### 14.3 自动质量评估

文件：`monitoring/evaluator.py`

评估维度：

```text
accuracy     40%
politeness   30%
completeness 30%
```

接口：

```python
evaluation = quality_evaluator.evaluate(
    question=req.message,
    answer=answer,
    context={
        "order_context": result.get("order_context"),
        "needs_human_transfer": result.get("needs_human_transfer", False),
        "transfer_reason": result.get("transfer_reason", ""),
        "user_memories": user_memories,
    },
)
quality_score = evaluation.score
```

低质量告警：

```python
if not evaluation.passed:
    self.trigger_alert(question=question, answer=answer, evaluation=evaluation)
```

当前评估器是确定性规则，不调用真实 LLM-as-judge。这样本地测试稳定，后续 M6 可以替换为评估数据集 + LLM 评审。

面试重点：

> M4 的价值不是“有一个分数”，而是把质量评估接入主链路：响应返回 `quality_score`，会话 metadata 保存评估明细，低于阈值产生告警事件，Prometheus 指标可被抓取。

---

## 15. M5 部署与压测：本地生产拓扑

M5 的目标是把本地服务从“单进程可运行”推进到“一键启动完整演示栈”。

### 15.1 Dockerfile 和 Compose 栈

核心文件：

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Compose 服务：

- `api`：FastAPI 应用，容器内使用 `uvicorn api.main:app --host 0.0.0.0 --port 8000`。
- `redis`：为 M3 `RateLimiter` 提供 Redis 计数后端。
- `postgres`：使用 `pgvector/pgvector:pg16`，承载业务会话/用户记忆表，并可承载 LangGraph checkpoint 表。
- `prometheus`：抓取 `api:8000/metrics`。
- `grafana`：自动加载 Prometheus datasource 和客服 Agent 看板。
- `locust`：通过 `loadtest` profile 启动压测入口。

面试重点：

> M5 先把 API、缓存、数据库、监控、看板和压测工具放进同一套可启动拓扑；存储升级后，Compose 默认已经使用 Postgres 保存业务会话、用户记忆和 LangGraph checkpoint。还没完成的是 Mem0/pgvector 语义记忆检索，而不是 Postgres 持久化本身。

### 15.2 Prometheus 和 Grafana provisioning

核心文件：

```text
infra/prometheus.yml
infra/grafana/provisioning/datasources/prometheus.yml
infra/grafana/provisioning/dashboards/customer-service.yml
infra/grafana/dashboards/customer-service.json
```

Grafana 看板覆盖：

- QPS by status。
- 平均响应时间。
- Token 使用。
- 活跃会话。
- 错误率。
- 转人工率。
- 平均质量分。

面试重点：

> M4 暴露指标，M5 负责把指标接入可视化。这样能说明系统不只是“有 `/metrics`”，而是有可运营的看板入口。

### 15.3 Locust 压测入口

文件：`load_tests/locustfile.py`

压测场景覆盖：

- 订单查询。
- 物流查询。
- 商品咨询。
- 退款申请和确认。
- 转人工。
- 健康检查。
- 指标抓取。

运行方式：

```bash
docker compose --profile loadtest up locust
```

或无界面压测：

```bash
docker compose --profile loadtest run --rm locust \
  -f /mnt/locust/locustfile.py \
  --host http://api:8000 \
  --headless -u 50 -r 5 -t 2m
```

面试重点：

> 当前 M5 提供压测入口和场景，不等于已经完成正式压测报告。正式报告需要在稳定 Docker 环境里运行，并记录平均延迟、P95、错误率和资源占用。

---

## 16. UI 层：为什么用 FastAPI 内置页面

文件：`api/ui.py`

当前 UI 是一个单文件 HTML：

- 左侧：用户 ID、会话 ID、快捷问题。
- 中间：聊天区和输入框。
- 右侧：质量分、Token、转人工状态、转人工原因、降级状态、用户记忆、会话摘要、订单上下文。

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

为什么不用 React/Vue？

- 当前目标是演示和调试 Agent 行为，不是复杂前端工程。
- 一个 FastAPI 服务即可提供 API 和演示 UI。
- UI 重点是可视化 `order_context`、`quality_score`、`token_used`、`degraded`、`user_memories` 等生产运营字段。

面试重点：

> 这个 UI 是工程演示台，不是产品级前端。它让面试官能直观看到客服回答、结构化上下文、转人工、记忆、降级和质量分，而不是只看接口 JSON。

---

## 17. 测试层：每组测试在保护什么

当前完整测试覆盖 M0-M6、存储后端切换和 checkpointer 装配。

### 17.1 `tests/test_chat_flow.py`

保护客服主流程：

- 订单状态。
- 物流状态。
- 商品库存。
- 退款二次确认。
- 转人工。
- 中英文混合输入。
- 空消息 422。
- 会话查询。
- 长期记忆保存、召回和删除。

### 17.2 `tests/test_memory.py`

保护 M2 记忆系统：

- 短期窗口裁剪和摘要。
- SQLite 会话持久化。
- 用户长期记忆保存、召回、删除。

### 17.3 `tests/test_storage_backends.py`

保护存储后端切换：

- 本地默认使用 SQLite 会话和用户记忆存储。
- `STORAGE_BACKEND=postgres` 时使用 Postgres 存储类。
- Postgres 会话存储保持现有会话接口契约。
- Postgres 用户记忆存储保持保存、召回、列出和删除接口契约。

### 17.4 `tests/test_checkpointer_factory.py`

保护 LangGraph checkpointer 装配：

- `CHECKPOINTER_BACKEND=none` 不注入 checkpointer。
- `CHECKPOINTER_BACKEND=postgres` 使用 `DATABASE_URL` 或 `CHECKPOINTER_URL` 并执行 `setup()`。
- `CHECKPOINTER_BACKEND=redis` 使用 `REDIS_URL` 或 `CHECKPOINTER_URL` 并执行 `setup()`。
- `CHECKPOINTER_SETUP=false` 可跳过自动初始化，适合生产迁移已完成的环境。

### 17.5 `tests/test_rate_limiter.py`

保护 M3 限流和预算：

- 同一用户第 11 次请求返回 `429`。
- 全局 QPS 超限返回 `503`。
- 全局 Token 预算超限返回降级 `200`。
- 单次 Token 预算超限返回降级 `200`。

### 17.6 `tests/test_llm_fallback.py`

保护 M3 模型弹性层：

- 主模型失败后切备用模型。
- 主模型失败会重试。
- 连续失败后熔断器打开。

### 17.7 `tests/test_observability.py`

保护 M4 可观测性：

- `/metrics` 暴露核心指标名。
- `/chat` 后请求、Token、延迟、质量分指标发生变化。
- 低质量回答触发告警事件。
- trace config 带 `session_id` 和 `user_id` metadata。

面试重点：

> 测试按层分布：API 流程、工具单元、记忆、存储后端、checkpointer、限流、模型弹性、观测和部署。这样比只做端到端测试更容易定位问题，也能证明每个生产能力都有独立验收标准。

---

### 17.8 `tests/test_deployment.py`

保护 M5 部署编排：

- Dockerfile 能启动 FastAPI 应用。
- Compose 定义 API、Redis、Postgres/pgvector、Prometheus、Grafana、Locust。
- Prometheus 抓取 `/metrics`。
- Grafana provisioning 包含 datasource 和 dashboard。
- Locust 覆盖核心客服压测场景。
- `.env.example` 不包含真实密钥。

## 18. 当前实现和设计稿的差异

这一节面试时很重要，因为它体现对项目状态诚实且清楚。

### 18.1 M6 如何接入 DeepSeek

当前 `/chat` 运行路径是：

```text
agent_node -> handle_customer_message -> intent/tools/FAQ-RAG -> ResilientLLM -> DeepSeek final answer
```

原因：

- 规则层负责客服安全边界和工具上下文，例如退款二次确认、投诉转人工和订单查询。
- 用户可见最终回答必须由真实 LLM 生成；DeepSeek 未配置或调用失败时 `/chat` 返回 `503 llm_unavailable`。
- `ResilientLLM` 统一承接 DeepSeek 主备模型、重试和熔断，Agent 节点不直接调用裸 provider。

### 18.2 当前已支持 LangGraph 原生 Checkpointer

虽然 `/chat` 传了：

```python
config={"configurable": {"thread_id": req.session_id}}
```

现在 `api/main.py` 会先调用 `build_checkpointer(settings)`，再把结果传给 `build_customer_service_graph(checkpointer=checkpointer)`：

```python
checkpointer = build_checkpointer(settings)
customer_service_graph = build_customer_service_graph(checkpointer=checkpointer)
```

配置含义：

- `CHECKPOINTER_BACKEND=none`：本地默认，不启用 LangGraph checkpoint。
- `CHECKPOINTER_BACKEND=postgres`：使用 `langgraph-checkpoint-postgres`，默认连接 `DATABASE_URL`。
- `CHECKPOINTER_BACKEND=redis`：使用 `langgraph-checkpoint-redis`，默认连接 `REDIS_URL`。
- `CHECKPOINTER_URL`：单独指定 checkpoint 连接串。
- `CHECKPOINTER_SETUP=true`：启动时执行 `setup()`；生产可由迁移流程预建表/索引后关闭。

注意：LangGraph checkpoint 和业务会话表职责不同。checkpoint 解决图状态恢复；`SessionStore` 解决客服业务查询、管理统计和用户可见会话材料。

### 18.3 当前工具不是 LangChain ToolNode

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

### 18.4 当前 Token 仍是轻量估算

当前：

- `token_used` 主要基于输入长度或上下文估算。
- Token 预算控制已经接入主链路。
- Prometheus 指标记录的是 `type="estimated"`。

后续接真实 LLM 后，应优先使用 provider 返回的 usage 数据。

面试讲法：

> 我会明确区分已实现和计划实现。当前 M6 已经完成客服闭环、记忆、限流、预算、弹性层、指标、质量评估、DeepSeek 必经主路径、FAQ/RAG、Docker Compose、Grafana provisioning、Locust 压测入口、100 题评测、Postgres 业务存储和 LangGraph Postgres/Redis checkpointer；但 ToolNode、Mem0/pgvector 语义记忆、正式压测报告和 24 小时长稳验证还在后续阶段。

---

## 19. 面试高频问题与回答

### Q1：为什么还保留规则型客服决策，而不是全部交给 LLM？

回答：

> 当前 `/chat` 已经要求最终回答必须走真实 LLM，但退款、投诉、法律和转人工这些高风险动作仍由规则层先固定安全边界。这样既能利用 LLM 生成自然话术，也不会让模型自由决定是否提交退款或忽略人工诉求。

### Q2：为什么要用 LangGraph，而不是 FastAPI 里一个函数写完？

回答：

> 因为客服 Agent 会演进成多节点状态流：记忆加载、上下文裁剪、LLM 推理、工具调用、质量评估、记忆保存、转人工。当前图虽然仍是线性三步，但 API 层已经围绕它接入了记忆、限流、预算和观测，后续扩展不用重写接口契约。

### Q3：退款为什么要二次确认？

回答：

> 退款属于有业务影响的动作，不能因为用户提到“退款”就直接提交。当前实现要求用户明确确认后才调用 `apply_refund(..., confirmed=True)`，这是高风险动作的安全边界。真实生产里还会结合权限、订单状态、风控和人工复核。

### Q4：转人工规则为什么优先级最高？

回答：

> 用户明确要求人工、投诉、法律、纠纷等场景继续让 AI 硬答会有业务风险。当前 `handle_customer_message()` 先判断转人工，再判断退款、物流、订单和商品，体现的是客服系统的安全优先级。

### Q5：为什么 Token 预算超限返回 200，而不是报错？

回答：

> Token 预算超限是可预期的生产情况，不是服务崩溃。返回降级 200 可以让用户得到明确提示，同时调用方通过 `degraded=true` 和 `degrade_reason` 知道本轮走了简化路径。

### Q6：M6 为什么禁用运行时离线模式？

回答：

> 05 项目现在用于演示真实 LLM 客服链路，所以运行时 `offline_stub` 会被拒绝。pytest 通过注入 fake LLM 保持稳定；真实运行必须配置 `LLM_MODE=deepseek` 和 `DEEPSEEK_API_KEY`。如果模型不可用，`/chat` 返回 `503 llm_unavailable`，避免把规则草稿当成真实模型回答。

### Q7：`quality_score` 现在怎么来的？

回答：

> M4 后 `quality_score` 来自 `AutoQualityEvaluator`。它按准确性 40%、礼貌性 30%、完整性 30% 计算加权分，低于阈值会记录告警事件，并把评估明细保存到会话 metadata。

### Q8：Prometheus 和 Grafana 现在做到什么程度？

回答：

> 当前实现了 Prometheus 兼容 `/metrics` 文本出口，并在 M5 通过 Docker Compose 编排 Prometheus 和 Grafana。Grafana 会自动加载 Prometheus datasource 和客服 Agent dashboard，展示 QPS、延迟、Token、错误率、转人工率和质量分。

### Q9：M2 的长期记忆和短期记忆有什么区别？

回答：

> 短期记忆服务于当前会话，重点是控制上下文窗口，所以它会摘要早期消息并保留最近 8 轮。长期记忆服务于跨会话用户画像，比如配送偏好和投诉记录，必须能保存、召回和删除。

### Q10：当前项目的最大短板是什么？

回答：

> 最大短板是还没有真实 LLM 主路径、正式压测报告、24 小时长稳验证和评估数据集。当前 M5 证明了本地可测、可编排的生产能力骨架，但不能宣称已经完成线上生产验证。

---

## 20. 面试时可以画的架构图

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
   | validate + rate limit + QPS + token budget
   v
Session / Memory Layer
   |
   | load session + load user memories + trim context
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
Quality + Observability
   |
   | evaluator + trace metadata + metrics
   v
ChatResponse
   |
   | answer + context + handoff + memories + quality + degraded
   v
SessionStore / UserMemoryManager / Metrics
   |
   | save session + extract memory + expose /metrics
   v
Browser UI

Docker Compose Runtime
   |
   | api + redis + postgres + prometheus + grafana + locust
   v
Monitoring / Load Test Demo
```

讲解重点：

- UI 和 API 在同一个 FastAPI 服务中。
- API 是生产边界，处理限流、预算、记忆、观测和持久化。
- LangGraph 当前是线性图，但为后续 ToolNode、真实 LLM 和质量节点留扩展空间。
- 业务工具返回结构化数据，不只是拼接字符串。
- 响应包含客服答案、结构化上下文和运营字段。
- M5 用 Docker Compose 把 API、Redis、Postgres、Prometheus、Grafana 和 Locust 放入同一个本地演示拓扑。

---

## 21. 你应该能现场演示什么

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

1. 输入：`我的订单 ORD123456 到哪了？`，观察订单状态和质量分。
2. 输入：`帮我查一下物流 ORD123456`，观察右侧订单上下文。
3. 输入：`AirBuds Pro 2 还有库存吗？`，说明商品查询不需要订单上下文。
4. 输入：`我要给订单 ORD123456 退款`，说明不会直接提交。
5. 输入：`我确认退款 ORD123456`，说明确认后才提交。
6. 输入：`我喜欢顺丰配送，以后发货优先顺丰`，再新会话问偏好，观察长期记忆召回。
7. 调 `DELETE /users/{user_id}/memories`，再召回一次，说明删除后不再返回旧记忆。
8. 输入投诉或人工诉求，观察转人工状态和原因。
9. 请求 `/metrics`，说明 Prometheus 可抓取指标。

接口演示：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","session_id":"session_001","message":"Where is my order ORD123456？谢谢"}'
```

指标演示：

```bash
curl http://127.0.0.1:8000/metrics
```

Compose 演示：

```bash
cp .env.example .env
docker compose up --build
```

监控入口：

```text
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
```

压测演示：

```bash
docker compose --profile loadtest run --rm locust \
  -f /mnt/locust/locustfile.py \
  --host http://api:8000 \
  --headless -u 50 -r 5 -t 2m
```

测试演示：

```bash
pytest tests -q
```

当前应通过 70 个测试。

---

## 22. 常见误区

### 误区 1：把当前项目说成已经完整生产可用

不准确。当前已完成 M6 的本地可测生产能力骨架、DeepSeek 主路径、FAQ/RAG 适配、管理接口、评测集和 Compose 编排，但真实线上生产验证还缺：

- 正式压测报告。
- 24 小时长稳验证。
- 真实外部告警系统。
- pgvector/Mem0 语义记忆迁移。

### 误区 2：把规则型决策说成 LLM Agent

当前不是让 LLM 自由执行所有业务动作，而是 LangGraph 编排下的规则 guardrail + DeepSeek 最终话术生成。更准确的说法是：

> 当前用规则型决策固定客服安全边界和工具上下文，最终客服话术必须通过 DeepSeek/真实 LLM 生成；同时接入记忆、限流、质量评估、指标、管理接口和自动评测。

### 误区 3：忽略降级路径

生产系统不能只讲成功路径。M3 已经覆盖：

- 用户限流返回 `429`。
- 全局 QPS 超限返回 `503`。
- Token 预算超限返回 `degraded=true` 的简化 `200`。

### 误区 4：忽略 `order_context`

`order_context` 是连接“自然语言客服回答”和“结构化业务系统”的关键。转人工、质检、记忆保存、UI 展示都可以复用它。

### 误区 5：把质量分当成绝对真理

M4 的质量评估器是确定性规则评估，不是最终真实评测体系。它的价值是把评估链路、告警和指标打通。M6 已引入 100 题评测集和自动评测报告，后续可以升级为 LLM-as-judge 或人工标注集。

---

## 23. 复盘总结

当前 05 项目的已开发部分可以总结为七句话：

1. M0 建好了可运行、可测试、可扩展的 FastAPI + LangGraph 骨架。
2. M1 跑通了客服核心闭环：订单、物流、商品、退款确认、转人工和 UI 演示。
3. M2 加入了短期记忆、会话状态和用户长期记忆，支持跨会话偏好召回和删除；当前本地默认 SQLite，Compose 默认 Postgres。
4. M3 加入了用户限流、全局 QPS、Token 预算降级和可测试的 LLM fallback/熔断层。
5. M4 加入了 trace metadata、Prometheus 兼容指标、自动质量评估和低质量告警。
6. M5 加入了 Docker Compose、Redis/Postgres/Prometheus/Grafana 编排、Grafana 看板和 Locust 压测入口，存储升级后 Compose 默认用 Postgres 承载业务存储和 LangGraph checkpoint。
7. M6 加入了 DeepSeek 主路径、FAQ/RAG 适配、管理接口、100 题评测集和自动评测报告。

面试收尾表达：

> 我在这个项目里关注的不是“让模型回答一句话”，而是把客服 Agent 做成可以持续演进的服务：有稳定 API、有显式状态、有工具边界、有安全规则、有记忆、有成本控制、有降级、有观测、有质量评估、有 DeepSeek 主路径、有 FAQ/RAG、有管理接口、有部署编排、有压测入口和 100 题自动评测基线。
