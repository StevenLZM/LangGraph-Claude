# 05_PRODUCT_AGENT 教学文档：生产级智能客服 Agent

这份文档用于带你系统学习 `05_PRODUCT_AGENT`。目标不是背目录，而是能从工程、Agent、生产稳定性和面试表达四个角度讲清楚这个项目。

读完后你应该能回答四类问题：

1. 这个项目解决什么真实业务问题。
2. 一次 `/chat` 请求在系统里如何流转。
3. 这个 Agent 为什么不是简单 Prompt，而是状态机、工具、安全边界、记忆和观测的组合。
4. 面试官追问生产级 Agent 时，应该讲哪些亮点、承认哪些边界。

---

## 1. 项目定位

`05_PRODUCT_AGENT` 是一个电商智能客服系统。它把大模型放进一个更接近生产环境的服务里，覆盖客服业务闭环、用户记忆、会话状态、限流、幂等、观测、评测、部署和业务消息。

一句话讲法：

> 这是一个基于 FastAPI 和 LangGraph 的生产级智能客服 Agent。它支持订单、物流、商品、退款和转人工场景，并在 API 边界加入 request_id 幂等、会话锁、乐观锁、三层记忆、检索决策、Token 预算、质量评估、Prometheus 指标、RocketMQ outbox、DeepSeek 真实 LLM 和 Docker Compose 部署。

面试时不要把它说成“已经可以直接大规模上线”。更准确的表述是：

> 当前是生产工程化教学样板，已经具备生产系统的关键结构，但真实上线还需要补齐认证授权、异步主路径、真实业务工具、独立 outbox relay、真实 embedding、长稳压测和合规审计。

---

## 2. 学习路线

建议按下面顺序读，而不是从所有文件随机开始。

### 第 1 步：先跑通项目

核心命令：

```bash
cd 05_PRODUCT_AGENT
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

常用接口：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","session_id":"session_001","request_id":"req_001","message":"我的订单 ORD123456 到哪了？"}'
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

注意：当前 `/chat` 主路径要求真实 LLM。默认配置是 `LLM_MODE=deepseek`，需要在 `.env` 中设置 `DEEPSEEK_API_KEY`。测试环境通过 `tests/conftest.py` 注入 fake LLM，所以 pytest 不需要真实 key。

### 第 2 步：先读 API 入口

阅读顺序：

```text
api/settings.py
api/schemas.py
api/main.py
api/idempotency.py
api/session_lock.py
api/middleware/rate_limiter.py
```

你要抓住一个核心事实：API 层不是薄薄转发到 Agent，它承担生产边界。

API 层负责：

- 请求协议校验。
- `request_id` 幂等。
- 同会话串行处理。
- 限流和全局 QPS。
- Token 预算降级。
- 会话、摘要记忆、长期记忆加载。
- LangGraph 调用。
- LLM 最终话术生成。
- 质量评估和指标记录。
- 会话持久化。
- RocketMQ 业务事件发布。

### 第 3 步：再读 Agent 图

阅读顺序：

```text
agent/graph.py
agent/state.py
agent/nodes.py
agent/dialog_state.py
agent/slot_filling.py
agent/tool_planner.py
agent/tool_executor.py
agent/response_builder.py
agent/tools.py
agent/retrieval.py
agent/intent.py
```

这个项目的 Agent 不是“完全交给 LLM 自由决定工具调用”。当前真实形态是：

```text
确定性多轮状态机
  + 规则/LLM 检索决策
  + 受控只读 ReAct 子循环
  + 写工具确认门
  + LLM 最终客服话术生成
```

这样设计的原因是客服系统有强业务边界。物流、订单、商品查询可以自动查；退款、退货等写操作必须先确认，不能让 LLM 自己决定直接执行。

### 第 4 步：理解三层记忆

阅读顺序：

```text
memory/short_term.py
memory/session_store.py
memory/summary.py
memory/long_term.py
memory/semantic.py
memory/factory.py
```

记忆分三层：

| 层次 | 作用 | 代码 | 范围 |
|---|---|---|---|
| 短期上下文 | 控制当前 Prompt 窗口，超长时摘要早期消息 | `ContextWindowManager` | 当前请求内 |
| 会话状态和摘要 | 保存 `session_id` 的历史消息、metadata、滚动摘要 | `SessionStore`、`SummaryMemoryStore` | 当前会话 |
| 用户长期记忆 | 保存用户偏好、投诉、姓名等跨会话信息 | `UserMemoryManager`、`SemanticMemoryStore` | 同一用户跨会话 |

面试要点：

> 业务会话存储和 LangGraph checkpointer 不是一回事。`SessionStore` 保存客服业务可见数据，支持 `/sessions`、管理接口和用户记忆治理；LangGraph checkpointer 保存图执行状态，用于恢复或追踪图运行。

### 第 5 步：读生产外围

阅读顺序：

```text
llm/factory.py
llm/resilient_llm.py
rag/faq_tool.py
monitoring/evaluator.py
monitoring/metrics.py
monitoring/tracing.py
messaging/events.py
messaging/outbox.py
messaging/publisher.py
messaging/handlers.py
docker-compose.yml
load_tests/locustfile.py
evals/run.py
```

这些模块回答的是“上线后怎么稳定运营”：

- LLM 不可用怎么办。
- FAQ/RAG 知识库不可用怎么办。
- 如何记录 QPS、延迟、Token 和质量分。
- 如何发布跨系统业务事件。
- 如何用 Docker Compose 拉起 API、Redis、Postgres、RocketMQ、Prometheus、Grafana。
- 如何用 Locust 压测。
- 如何用评测集做回归。

---

## 3. 当前目录速查

```text
05_PRODUCT_AGENT/
├── api/                    # FastAPI 入口、请求协议、幂等、会话锁、限流、管理接口、UI
├── agent/                  # LangGraph 图、多轮状态机、检索决策、工具计划、工具执行、回复构造
├── memory/                 # 短期上下文、会话存储、摘要记忆、长期记忆、语义记忆
├── llm/                    # DeepSeek/OpenAI-compatible/Anthropic 客户端工厂、重试、备用模型、熔断
├── rag/                    # FAQ/RAG 工具，适配兄弟项目 01_RAG
├── messaging/              # RocketMQ event envelope、outbox、publisher、后处理 handler
├── monitoring/             # Prometheus 指标、LangSmith metadata、自动质量评估
├── evals/                  # 100 题评测集、评测脚本、报告生成
├── load_tests/             # Locust 压测脚本
├── infra/                  # Prometheus 和 Grafana 配置
├── tests/                  # pytest 覆盖核心链路
├── docker-compose.yml      # API、Redis、Postgres、RocketMQ、Prometheus、Grafana、Locust 编排
├── Dockerfile
├── README.md
├── DEV_PROGRESS.md
├── PRODUCTION_READINESS_GAPS.md
└── Codex_TEACHING_GUIDE.md
```

---

## 4. 技术栈总览

| 类别 | 技术 | 项目中的作用 |
|---|---|---|
| API | FastAPI、Pydantic、uvicorn | 暴露 `/chat`、`/health`、`/metrics`、管理接口和内置客服 UI |
| Agent 编排 | LangGraph | 编排客服多轮状态机，支持可选 checkpointer |
| LLM | DeepSeek OpenAI-compatible、LangChain ChatOpenAI、ChatAnthropic | 真实 LLM 主路径和可选其他 provider |
| LLM 弹性 | 自研 `ResilientLLM` | 主备模型、指数退避、熔断器 |
| 记忆 | SQLite、Postgres、Milvus Lite 可选 | 会话、摘要、长期记忆、语义记忆 |
| 并发控制 | Redis、内存 fallback | 用户限流、全局 QPS、同会话分布式锁 |
| 持久化 | SQLite、本地文件、Postgres | 本地演示和生产演示两套后端 |
| 消息 | RocketMQ、Outbox pattern | 发送客服完成、后处理、转人工提醒事件 |
| 观测 | Prometheus text exposition、LangSmith metadata、Grafana | 指标、链路标签、仪表盘 |
| 评测 | pytest、自定义 eval runner、Locust | 单元测试、自动问答评测、压测入口 |
| 部署 | Docker、Docker Compose | 本地一键拉起生产依赖 |

---

## 5. 一次 `/chat` 请求怎么走

以“我的订单 ORD123456 到哪了？”为例：

```text
用户请求
  -> ChatRequest 校验 user_id / session_id / request_id / message
  -> ChatRequestStore.start_request 写入 processing 幂等记录
  -> 重复 request_id 返回首次响应，冲突 message 返回 409
  -> SessionLockManager 获取 session_id 会话锁
  -> RateLimiter 检查用户限流和全局 QPS
  -> SessionStore 加载历史消息、metadata、version
  -> SummaryMemoryStore 加载会话摘要
  -> decide_retrieval 判断是否需要长期记忆、FAQ/RAG、业务上下文
  -> UserMemoryManager / SemanticMemoryStore 召回长期记忆
  -> ContextWindowManager 裁剪上下文并估算 token
  -> RateLimiter.reserve_token_budget 检查 Token 预算
  -> build_trace_config 注入 LangSmith tags 和 metadata
  -> customer_service_graph.invoke 调用 LangGraph
  -> 图内执行路由、补槽、确认、工具规划、工具执行、回复构造
  -> _generate_required_llm_answer 用真实 LLM 生成最终客服话术
  -> AutoQualityEvaluator 评估质量
  -> UserMemoryManager 保存可提取的长期记忆
  -> SessionStore.save_session 用 expected_version 保存会话
  -> record_chat_request 写 Prometheus 兼容指标
  -> message_publisher.publish_many 写 outbox 并尝试发 RocketMQ
  -> ChatRequestStore.complete_success 保存首次响应快照
  -> 释放会话锁
  -> 返回 ChatResponse
```

代码映射：

| 环节 | 关键文件 | 关键点 |
|---|---|---|
| 请求模型 | `api/schemas.py` | `ChatRequest` 必须带 `request_id` |
| 幂等 | `api/idempotency.py` | 主键是 `(user_id, session_id, request_id)`，message hash 防复用 |
| 会话锁 | `api/session_lock.py` | Redis `SET NX PX`，无 Redis 时内存锁 |
| 限流 | `api/middleware/rate_limiter.py` | 用户分钟级限流、全局 QPS、Token 预算 |
| 会话存储 | `memory/session_store.py` | `expected_version` 乐观锁防覆盖 |
| 检索决策 | `agent/retrieval.py` | rules 默认，LLM JSON 可选，失败回退规则 |
| 图执行 | `agent/graph.py`、`agent/nodes.py` | 多轮状态机主图 |
| 工具 | `agent/tools.py` | Mock 订单、物流、商品、退款 |
| 回复 | `agent/response_builder.py`、`api/main.py` | 规则草稿 + LLM 最终话术 |
| 观测 | `monitoring/*` | 指标、质量分、trace metadata |
| 消息 | `messaging/*` | Outbox + RocketMQ 事件 |

---

## 6. API 层重点

`api/main.py` 启动时构建一组进程级对象：

```python
checkpointer = build_checkpointer(settings)
customer_service_graph = build_customer_service_graph(checkpointer=checkpointer)
context_window_manager = ContextWindowManager()
session_store = build_session_store(settings)
chat_request_store = build_chat_request_store(settings)
session_lock_manager = SessionLockManager(redis_url=settings.redis_url)
user_memory_manager = build_user_memory_manager(settings)
summary_memory_store = build_summary_memory_store(settings)
semantic_memory_store = build_semantic_memory_store(settings)
quality_evaluator = AutoQualityEvaluator(...)
customer_service_llm_setup = build_customer_service_llm(settings)
message_outbox_store = build_message_outbox_store(settings)
message_publisher = build_message_publisher(settings, outbox_store=message_outbox_store)
rate_limiter = RateLimiter(...)
```

这说明：

- 图和存储组件不会每次请求重复创建。
- 本地默认 SQLite、内存限流、内存会话锁、无 checkpointer。
- Docker Compose 默认 Postgres、Redis、Postgres checkpointer、RocketMQ。
- LLM 是必需主路径，不再静默 fallback 到离线规则答案。

核心接口：

```text
GET  /
GET  /health
GET  /metrics
POST /chat
GET  /chat/requests/{request_id}
GET  /sessions/{session_id}
DELETE /users/{user_id}/memories
GET  /admin/sessions
GET  /admin/users/{user_id}/memories
GET  /admin/stats/transfers
GET  /admin/messages/outbox
```

面试讲法：

> `/chat` 是有副作用的写接口。它会调用 LLM、写会话、写记忆、写指标、发消息，所以必须在 API 边界处理幂等、同会话并发和版本冲突。

---

## 7. 幂等、锁和一致性

### request_id 幂等

客户端每发送一条新消息生成一个新 `request_id`。如果网络超时或浏览器重试，必须复用同一个 `request_id`。

服务端逻辑：

```text
第一次请求：
  start_request 插入 processing
  正常处理
  complete_success 保存 response_json

重复请求：
  start_request 插入失败
  message_hash 一致
  如果 succeeded，直接返回首次 response_json，request_status=replayed
  如果 processing，短暂等待，仍未完成返回 202
  如果 failed，返回首次失败

冲突请求：
  同一个 request_id 但 message_hash 不一致，返回 409
```

面试追问：

> 为什么不用 message 文本直接去重？

回答：

> 文本相同不代表同一条业务请求。幂等必须由客户端明确声明逻辑请求 ID。服务端用 message hash 只是防止同一个 `request_id` 被错误复用到不同消息。

### 同会话并发

`request_id` 只解决“同一条消息重复提交”，不能解决“同一会话两条不同消息同时提交”。

所以项目有两层保护：

```text
SessionLockManager
  Redis: SET chat:session_lock:{session_id} token NX PX ttl
  Memory fallback: asyncio.Lock

SessionStore.save_session(expected_version=N)
  数据库 WHERE session_id=? AND version=?
  版本不匹配抛 SessionVersionConflict
```

面试讲法：

> Redis 锁减少同一会话并发进入 LLM 和工具链路的概率，数据库乐观锁是最后一致性保护。即使锁过期、实例异常或未来出现旁路写入，也不会静默覆盖旧会话。

---

## 8. Agent 图结构

`agent/graph.py` 中的 LangGraph 主图：

```text
START
  -> context_loader
  -> turn_router
  -> pending_task_resolver | retrieval_decision | response_builder
  -> slot_filling
  -> confirmation_guard
  -> tool_planner_or_react
  -> tool_executor
  -> tool_planner_or_react | response_builder
  -> finalizer
  -> END
```

每个节点职责：

| 节点 | 作用 |
|---|---|
| `context_loader` | 初始化 state，提取最新用户消息，归一化 dialog_state |
| `turn_router` | 判断新任务、继续任务、确认、取消、过期、转人工 |
| `pending_task_resolver` | 处理上一轮挂起任务，如确认、取消、过期 |
| `retrieval_decision` | 缺省时执行规则检索决策 |
| `slot_filling` | 抽取订单号、退货原因、商品状态等槽位 |
| `confirmation_guard` | 拦截退款等写操作，未确认时不进入工具执行 |
| `tool_planner_or_react` | 构造工具计划，或继续只读 ReAct 子循环 |
| `tool_executor` | 执行下一步工具，记录 `tool_trace` |
| `response_builder` | 构造规则草稿、choices、task_status |
| `finalizer` | 统计窗口大小、轮次和耗时 |

核心状态在 `agent/state.py`：

```text
session_id / user_id
messages
user_memories
memory_summary
retrieval_decision
dialog_state
task_status
tool_plan
tool_results
tool_trace
order_context
choices
needs_human_transfer
transfer_reason
token_used
response_time_ms
```

面试讲法：

> 这个 Agent 把不确定性控制在合适位置：任务推进、补槽、确认、工具权限由确定性节点控制，最终话术由 LLM 根据后端上下文生成。这样既能利用 LLM 的表达能力，也能避免 LLM 越权执行退款等写操作。

---

## 9. dialog_state 和多轮任务

`dialog_state` 是这个项目从“单轮规则客服”走向“多轮 Agent”的关键。

它包含：

```text
active_task        当前任务，如 refund / logistics / product
phase              collecting_slots / awaiting_confirmation / executing / completed / cancelled / escalated
required_slots     必填槽位
collected_slots    已收集槽位
missing_slots      缺失槽位
confirmation       是否需要确认、是否已确认、动作名、预览文案
attempt_count      补槽尝试次数
expires_at         任务过期时间
idempotency_key    写工具业务幂等键
```

典型退款流程：

```text
用户：我要退货
  -> active_task=refund
  -> phase=collecting_slots
  -> missing_slots=["order_id", "reason", "product_condition"]

用户：ORD123456
  -> 收集 order_id
  -> 继续缺 reason / product_condition

用户：不想要了，商品完好
  -> phase=awaiting_confirmation
  -> response_builder 返回 choices
  -> 不执行 apply_refund

用户：确认退货
  -> phase=executing
  -> confirmation_guard 放行
  -> tool_executor 调用 apply_refund
  -> phase=completed
```

面试追问：

> 为什么不用历史聊天记录让 LLM 自己判断当前是不是确认退款？

回答：

> 退款是写操作，不能只靠自然语言历史推断。显式状态能记录缺哪些槽位、是否已确认、是否过期、重试次数和幂等键。这样测试、审计和恢复都更可靠。

---

## 10. 工具系统和安全边界

当前工具在 `agent/tools.py`：

```text
get_order       查询订单
get_logistics   查询物流
get_product     查询商品
apply_refund    提交退款申请
```

`tool_planner.py` 会根据任务和意图生成工具计划：

- 订单查询：`get_order`
- 物流查询：`get_order -> get_logistics`
- 商品咨询：`get_product`
- FAQ：`faq_rag`
- 退款资格只读问题：`get_order -> get_logistics -> faq_rag`
- 确认后的退款写操作：`apply_refund`

关键安全规则：

1. 只读工具可以自动执行。
2. 只读 ReAct 最多 3 步。
3. `apply_refund` 必须经过 `confirmation_guard`。
4. `apply_refund` 必须携带 `dialog_state.idempotency_key`。
5. `tool_executor` 对外隐藏 `idempotency_key`，避免泄漏内部幂等细节。

面试讲法：

> Tool use 不是让模型随便调函数，而是要给工具分级。查询类工具可以自动化，写操作必须有确认、权限、幂等和审计。这个项目用确定性确认门先把边界立住，再让 LLM 负责表达。

---

## 11. 检索决策和 RAG

`agent/retrieval.py` 负责判断本轮是否需要外部上下文。

输出结构：

```text
intent
needs_memory
needs_faq
needs_business_context
memory_filters
external_sources
query_rewrite
confidence
decision_source
error
```

默认模式：

```text
RETRIEVAL_DECISION_MODE=rules
```

可选模式：

```text
RETRIEVAL_DECISION_MODE=llm
```

LLM 模式要求模型输出 JSON。解析失败或调用失败时退回规则决策。

FAQ/RAG 在 `rag/faq_tool.py`：

- 尝试加载兄弟项目 `01_RAG` 的 hybrid retriever。
- 知识库不可用时不会让 `/chat` 崩溃。
- 返回 `matched`、`answer`、`sources`、`backend`、`error`。

面试讲法：

> 检索决策是为了控制上下文成本和外部依赖。不是每轮都查长期记忆、RAG 和业务工具，而是先判断这轮需要什么。这样能减少无效检索、降低延迟，也能让失败降级更明确。

---

## 12. 记忆系统

### 短期上下文

`ContextWindowManager` 做两件事：

- 保留最近消息。
- 超长时把早期消息压成 `SystemMessage(type=summary)`。

当前 token 估算是轻量实现：

```python
max(1, len(text) // 2)
```

它适合教学和测试，但不是严格生产 token 计数。生产应接入模型 provider usage 或 tokenizer。

### 会话存储

`SessionStore` 按 `session_id` 保存：

- messages
- metadata
- version
- updated_at

metadata 中包含：

```text
summary
needs_human_transfer
transfer_reason
token_used
quality_score
quality_evaluation
quality_alert
trace_metadata
llm_trace
retrieval_decision
dialog_state
task_status
tool_trace
```

### 摘要记忆

`SummaryMemoryStore` 按 `session_id` 保存滚动摘要：

```text
session_id
user_id
summary
version
covered_turns
source_event_id
updated_at
```

`messaging/handlers.py` 的 `PostprocessEventHandler` 可以在后处理事件里更新摘要。

### 长期关键词记忆

`UserMemoryManager` 会从用户消息中提取：

- 配送偏好。
- 普通偏好。
- 投诉记录。
- 用户姓名。

例如：

```text
我喜欢顺丰配送，以后发货优先顺丰
  -> 用户偏好：我喜欢顺丰配送，以后发货优先顺丰
```

配送偏好会替换旧偏好，避免“顺丰”和“京东物流”同时存在导致回答冲突。

### 语义长期记忆

`SemanticMemoryStore` 有三种实现：

```text
NoopSemanticMemoryStore
SQLiteSemanticMemoryStore
MilvusSemanticMemoryStore
```

SQLite 版本用规则打分。Milvus 版本当前使用 `_hash_embedding()` 生成向量，这是教学 fallback，不是真实 embedding。

面试讲法：

> 项目已经把语义记忆接口、数据结构、Milvus backend 和删除接口打通，但召回质量还不是生产级。生产需要真实 embedding、模型版本、记忆抽取评测、冲突合并和隐私治理。

---

## 13. LLM 主路径和降级

`llm/factory.py` 负责构建 LLM：

- 默认 `LLM_MODE=deepseek`。
- DeepSeek 通过 `ChatOpenAI` 的 OpenAI-compatible 接口调用。
- 主模型是 `DEEPSEEK_MAX_MODEL`。
- 备用模型是 `DEEPSEEK_LIGHT_MODEL`。
- `offline_stub` 已禁用，不能在 `/chat` 中静默返回规则答案。

`ResilientLLM` 提供：

- 主模型调用。
- 失败后指数退避重试。
- 达到阈值后熔断。
- 熔断打开时直接走备用模型。
- 返回 `model_used`、`fallback_used`、`attempts`、`circuit_state`。

`api/main.py` 中图先生成规则草稿，然后 `_generate_required_llm_answer()` 用真实 LLM 基于后端上下文生成最终回答。

系统提示词强调：

- 只能基于后端工具结果回答。
- 不能编造订单、物流或退款信息。
- 有 choices 时不能新增未提供选项。
- 没有工具执行时不能声称已退款、改地址或取消订单。
- `task_status=awaiting_confirmation` 时只能请求确认。

面试追问：

> 为什么不是让 LLM 直接生成完整答案？

回答：

> 客服里的事实必须来自业务系统，不能来自模型想象。所以先由状态机和工具拿到结构化上下文，再让 LLM 做受约束的表达。这样可测试、可审计，也能减少幻觉。

---

## 14. 观测、质量和评测

### Prometheus 指标

`monitoring/metrics.py` 暴露：

```text
agent_requests_total{status=...}
agent_response_time_seconds_count
agent_response_time_seconds_sum
agent_tokens_total{type="estimated"}
agent_active_sessions
agent_quality_score_count
agent_quality_score_sum
agent_errors_total
agent_human_transfers_total
```

访问：

```bash
curl http://127.0.0.1:8000/metrics
```

### LangSmith metadata

`monitoring/tracing.py` 注入：

```text
tags = ["customer-service", "session:{session_id}", "user:{user_id}"]
metadata = {
  "session_id": ...,
  "user_id": ...,
  "environment": ...,
  "version": ...
}
```

只有 LangGraph/LLM 主调用会产生 trace。`/health` 和 `/metrics` 不会产生对话 trace。

### 自动质量评估

`AutoQualityEvaluator` 按三项打分：

```text
准确性 40%
礼貌性 30%
完整性 30%
```

低于阈值时记录 warning 事件，并在会话 metadata 中标记 `quality_alert=true`。

### 自动评测

`evals/run.py` 支持：

```bash
python evals/run.py --dataset evals/dataset.jsonl --dry-run
python evals/run.py --dataset evals/dataset.jsonl
```

评测结果写入：

```text
evals/results/<run_id>/results.jsonl
evals/results/<run_id>/REPORT.md
```

面试讲法：

> 生产级 Agent 不能只看“能回答”。还要看质量分、转人工率、Token 成本、错误率、延迟和评测集回归。这个项目用 Prometheus 记录运营指标，用自动评估做质量信号，用 eval dataset 做离线回归。

---

## 15. RocketMQ 和 Outbox

`/chat` 成功后会构造事件：

| 事件 | Topic | 用途 |
|---|---|---|
| `customer_service.chat_completed` | `agent-customer-service-normal-v1` | 对话完成事件 |
| `customer_service.postprocess_requested` | `agent-customer-service-fifo-v1` | 会话内有序后处理 |
| `customer_service.human_transfer_reminder_requested` | `agent-customer-service-delay-v1` | 转人工延迟提醒 |

事件统一 envelope：

```text
event_id
event_type
event_version
producer
occurred_at
trace_id
aggregate_id
payload
```

Outbox 逻辑：

```text
enqueue(event) 写 message_outbox，状态 pending
producer.send 成功，mark_published
producer.send 失败，mark_failed，状态 enqueue_failed
```

当前边界：

- 已有 outbox 表和失败可见性。
- 已有 `PostprocessEventHandler`。
- 但没有独立 relay worker、死信队列、自动重试和重放闭环。

面试讲法：

> 现在是 outbox pattern 的教学实现。请求内先写 outbox 再尝试发送 RocketMQ，发送失败不会阻断 `/chat`，但完整生产化还需要独立 relay worker 扫描 pending/enqueue_failed，并做指数退避、死信和人工重放。

---

## 16. 部署和运行

### 本地 Python 运行

```bash
cd 05_PRODUCT_AGENT
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
cd 05_PRODUCT_AGENT
cp .env.example .env
docker compose up --build
```

Compose 会启动：

```text
api
redis
postgres/pgvector
rocketmq-namesrv
rocketmq-broker
prometheus
grafana
locust profile
```

访问：

```text
API:        http://127.0.0.1:8000
UI:         http://127.0.0.1:8000/
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
Locust:     http://127.0.0.1:8089
```

压测入口：

```bash
docker compose --profile loadtest up locust
```

---

## 17. 测试怎么读

核心测试文件：

| 测试 | 覆盖点 |
|---|---|
| `test_chat_flow.py` | `/chat` 订单、物流、商品、退款、多轮、记忆、真实 LLM 注入 |
| `test_graph_skeleton.py` | LangGraph 节点、补槽、确认门、只读 ReAct、写工具幂等键 |
| `test_chat_idempotency.py` | request_id 重放、状态查询、冲突检测 |
| `test_memory.py` | 短期窗口、会话持久化、乐观锁、长期记忆 |
| `test_retrieval_decision.py` | 规则/LLM 检索决策和 fallback |
| `test_retrieval_integration.py` | 语义记忆召回进入 `/chat` |
| `test_rate_limiter.py` | 用户限流、全局 QPS、Token 预算降级 |
| `test_llm_factory.py` | DeepSeek/OpenAI-compatible 构建和启动错误 |
| `test_llm_fallback.py` | 重试、备用模型、熔断 |
| `test_rocketmq_messaging.py` | 事件 envelope、outbox、后处理幂等 |
| `test_observability.py` | 指标和质量评估 |
| `test_storage_backends.py` | SQLite/Postgres 工厂和契约 |
| `test_deployment.py` | Docker/Compose/infra 配置存在性 |
| `test_evals.py` | 自动评测 pipeline |

运行：

```bash
cd 05_PRODUCT_AGENT
pytest tests -q
pytest tests/test_graph_skeleton.py -q
pytest tests/test_chat_flow.py -q
```

---

## 18. 面试重点总览

### 重点 1：项目不是 Demo，而是生产链路教学样板

讲法：

> 我做的不是只会回复的 Chatbot，而是一个有生产边界的智能客服服务。一次请求会经过幂等、锁、限流、记忆、检索决策、Agent 状态机、工具执行、LLM 最终生成、质量评估、指标、消息 outbox 和持久化。

面试官可能追问：

> 生产级体现在哪里？

回答要点：

- `request_id` 幂等防客户端重试重复扣费和重复消息。
- 同 session 锁和 DB 乐观锁防会话覆盖。
- Redis/内存 hybrid 限流。
- Token 预算超限降级。
- SQLite/Postgres 存储后端切换。
- LangGraph Postgres/Redis checkpointer 可选。
- Prometheus、LangSmith metadata、质量评估。
- RocketMQ outbox 事件。
- Docker Compose、Locust、自动评测。

### 重点 2：Agent 架构是受控的

讲法：

> 这个 Agent 不是完全开放的 autonomous agent，而是确定性多轮状态机加受控只读 ReAct。原因是客服有退款、投诉、转人工等安全边界，不能让 LLM 自由决定写操作。

面试官可能追问：

> 为什么不直接用 LangChain tool calling？

回答要点：

- 客服写操作需要确认、权限、幂等、审计。
- Tool-calling 可以作为下一步演进，但 guardrail 必须在后端。
- 当前先用状态机保证核心业务正确性，再用 LLM 生成客服话术。

### 重点 3：记忆不是一个 messages 列表

讲法：

> 项目把记忆拆成短期上下文、会话摘要、用户长期记忆和语义长期记忆。短期记忆解决上下文窗口，摘要解决长会话，长期记忆解决跨会话偏好，语义记忆提供可替换的 Milvus 接口。

面试官可能追问：

> 用户要求删除记忆怎么做？

回答要点：

- `DELETE /users/{user_id}/memories` 删除关键词长期记忆。
- 同时调用 `semantic_memory_store.delete_user(user_id)` 删除语义记忆。
- 生产还要增加认证、审计、租户隔离和 PII 脱敏。

### 重点 4：幂等和会话一致性

讲法：

> `/chat` 是写接口，必须处理重复提交和同会话并发。项目用 `request_id` 做逻辑请求幂等，用 Redis 会话锁串行化同 session 请求，用 session version 乐观锁防止最终写覆盖。

面试官可能追问：

> 如果请求处理到一半进程挂了怎么办？

回答要点：

- 当前幂等表可能停在 `processing`，重复请求等待后返回 `202 request_processing`。
- 这是已识别的生产差距。
- 生产应增加 processing 超时回收、失败恢复任务和请求状态机清理策略。

### 重点 5：RAG 和检索决策

讲法：

> 项目没有每轮盲目查 RAG，而是先做检索决策，判断是否需要长期记忆、FAQ/RAG 和业务上下文。默认规则决策稳定可测，LLM JSON 决策可选，失败退回规则。

面试官可能追问：

> RAG 不可用时会怎样？

回答要点：

- `FAQRAGTool` 捕获加载和检索异常。
- 返回 `backend=unavailable` 和明确未命中答案。
- `/chat` 不因知识库不可用而崩溃。

### 重点 6：LLM 失败处理

讲法：

> 当前正常 `/chat` 不允许静默退回离线规则答案。DeepSeek key 未配置或调用失败时返回 `503 llm_unavailable`，避免用户以为系统给出了真实模型能力的回答。

面试官可能追问：

> 那 `ResilientLLM` 的作用是什么？

回答要点：

- 对真实 provider 做主模型重试。
- 多次失败后走 fallback model。
- 熔断打开时短期内跳过主模型。
- 返回 trace 字段用于观测。

### 重点 7：Outbox 和消息最终一致性

讲法：

> 对话成功后会写业务事件到 outbox，再尝试发 RocketMQ。发送失败只标记 `enqueue_failed`，不阻塞用户响应。

面试官可能追问：

> 这算完整 outbox 吗？

回答要点：

- 目前是请求内 outbox 和失败可观测。
- 完整生产还需要独立 relay worker、重试退避、死信、重放和多 worker 并发控制。

### 重点 8：生产差距要主动说

不要回避当前问题。可以这样说：

> 这个项目已经具备生产形态，但仍有几个 P0 差距：Admin 和用户接口缺认证授权，`/chat` async 入口里仍有同步图和同步存储调用，RocketMQ outbox 缺独立 relay，业务工具还是 mock，语义记忆用 hash embedding，健康检查还不是依赖真实探测。

这个表达比“已经生产可用”更专业。

---

## 19. 面试官高频追问和参考回答

### Q1：这个项目和普通 Chatbot 最大区别是什么？

参考回答：

> 普通 Chatbot 主要关注 Prompt 和模型回复，这个项目关注生产客服链路。它处理请求幂等、会话并发、长期记忆、检索决策、工具安全、转人工、质量评估、指标、消息 outbox 和部署。LLM 是其中一环，不是整个系统。

### Q2：为什么需要 LangGraph？

参考回答：

> 客服流程天然有状态，例如退款要先收集订单号和原因，再要求确认，最后才能执行写工具。LangGraph 适合把这些状态节点、条件边和可恢复上下文表达清楚。相比单个函数或纯 Prompt，图结构更容易测试和扩展。

### Q3：退款为什么不能直接由 LLM 调用工具？

参考回答：

> 退款是写操作，可能造成资金和工单影响。必须经过后端确认门、显式 dialog_state、幂等键和可审计工具调用。LLM 可以生成解释话术，但不能绕过业务 guardrail。

### Q4：为什么需要 `request_id`，HTTP 重试不是客户端自己处理吗？

参考回答：

> 客户端只能重试，但服务端必须知道两次请求是不是同一条逻辑消息。否则超时后重试可能重复调用 LLM、重复提交退款或重复发布事件。`request_id` 是客户端和服务端共同约定的幂等语义。

### Q5：Redis 锁和数据库乐观锁为什么两个都要？

参考回答：

> Redis 锁让同一会话请求尽量串行进入核心链路，减少竞态和重复 LLM 调用。数据库乐观锁是最终保护，防止锁失效、超时或旁路写入导致会话覆盖。一个偏运行时协调，一个偏持久化一致性。

### Q6：记忆如何避免污染？

参考回答：

> 当前实现先做了基础限制：疑问句不写记忆，偏好类按关键词提取，配送偏好会替换旧值。生产环境还要增加结构化 LLM 抽取、敏感信息过滤、置信度、过期策略、用户可编辑和召回评测。

### Q7：如何评估 Agent 回答质量？

参考回答：

> 在线侧记录质量分、转人工率、Token、延迟和错误率；离线侧使用 `evals/dataset.jsonl` 做回归评测。当前自动质量评估是规则评分，能做运营信号，但不能替代人工标注和更严格的 LLM-as-judge 或业务验收。

### Q8：如果 DeepSeek 不可用，系统怎么办？

参考回答：

> 配置缺失时 `/chat` 返回 `503 llm_unavailable`，不静默回退规则答案。调用失败时 `ResilientLLM` 可做重试、fallback 和熔断。如果完全不可用，API 返回明确错误，运维可以通过指标和日志定位。

### Q9：为什么 FAQ/RAG 只是工具之一？

参考回答：

> 客服问题不都需要知识库。订单和物流需要业务工具，偏好问题需要长期记忆，退款需要状态机和确认。RAG 只适合政策、FAQ、售后规则等知识类问题，所以前面有检索决策控制是否调用。

### Q10：这个项目下一步怎么生产化？

参考回答：

> 我会优先做 P0：认证授权和租户隔离、主路径异步化、Postgres 连接池、独立 outbox relay、真实业务工具 adapter、真实 embedding 和记忆治理、依赖健康检查、长稳压测和告警闭环。

### Q11：请你现场画一下 `/chat` 的核心链路

考察点：候选人是否真的理解系统，而不是只背技术栈。

我期望候选人这样回答：

> `/chat` 先做 Pydantic 请求校验，然后用 `(user_id, session_id, request_id)` 写幂等记录。拿到处理权后获取同 `session_id` 的会话锁，再做用户限流、全局 QPS 和 Token 预算检查。接着加载 SessionStore 的历史消息、SummaryMemoryStore 的会话摘要，执行检索决策，按决策读取长期记忆和语义记忆，然后裁剪上下文进入 LangGraph。图里通过 `turn_router`、`slot_filling`、`confirmation_guard`、`tool_planner_or_react`、`tool_executor` 和 `response_builder` 推进任务，之后用真实 LLM 基于工具结果生成最终话术。最后做质量评估、保存会话、写指标、发布 RocketMQ outbox 事件、完成幂等记录并释放锁。

容易扣分的回答：

- 只说“FastAPI 调 LangGraph，然后 LangGraph 调 LLM”。
- 说不清幂等、会话锁、记忆和质量评估在哪一层。
- 把业务事实说成由 LLM 直接生成。

### Q12：如果两个请求同时打到同一个 `session_id`，系统如何处理？

考察点：并发一致性。

我期望候选人这样回答：

> 同一会话并发不是靠 `request_id` 解决的。`request_id` 只保证同一条逻辑消息重复提交不会重复处理。同一 `session_id` 的两条不同消息要靠 `SessionLockManager` 串行化，生产配置下用 Redis `SET NX PX`，本地测试用 `asyncio.Lock`。保存会话时再用 `expected_version` 做数据库乐观锁，如果加载的版本已经过期，就抛 `SessionVersionConflict` 并返回 409。这是运行时锁加持久化版本保护的双层设计。

容易扣分的回答：

- 把 `request_id` 和同会话并发混为一谈。
- 只说“用了 Redis 所以安全”，不提锁租约失效后的 DB 版本保护。

### Q13：为什么 Token 预算超限返回降级回答，而不是直接报错？

考察点：成本控制和用户体验。

我期望候选人这样回答：

> Token 预算超限是预期内的容量和成本控制场景，不等价于系统异常。项目里单次预算或全局小时预算超限时，会返回 `degraded=true` 和 `degrade_reason`，同时给用户一个简化回复。这让调用方知道本轮是预算降级，不会误判为正常完整回答；也避免因为成本保护直接让用户看见 500。生产里还应把真实 provider usage 回写到预算系统，而不是只靠当前轻量估算。

容易扣分的回答：

- 只说“为了不报错”。
- 不提 `degraded`、`degrade_reason` 和预算观测。
- 不承认当前 token 估算还不够精确。

### Q14：`SessionStore` 和 LangGraph checkpointer 有什么区别？

考察点：是否理解业务状态和图执行状态的边界。

我期望候选人这样回答：

> `SessionStore` 是业务存储，保存用户可见的消息、metadata、质量分、转人工状态、`dialog_state`、`task_status` 等，支撑 `/sessions`、管理接口和会话审计。LangGraph checkpointer 是图执行状态存储，用于 LangGraph 原生 checkpoint/resume。两者可以都落在 Postgres，但职责不同，不能用 checkpoint 表替代业务会话表，也不能把用户隐私删除只理解成清 checkpoint。

容易扣分的回答：

- 说“都是存对话历史”。
- 认为接了 checkpointer 就不需要 `SessionStore`。

### Q15：这个项目为什么叫“受控只读 ReAct”，不是完整 ReAct Agent？

考察点：Agent 安全边界。

我期望候选人这样回答：

> 当前 ReAct 只用于只读复合问题，例如退款资格和物流追问可以按 `get_order -> get_logistics -> faq_rag` 最多执行 3 步。写工具如 `apply_refund` 不进入开放循环，必须先由 `confirmation_guard` 检查 `dialog_state` 里的确认状态，再携带幂等键执行。这样做牺牲了一些开放性，但换来业务安全、可测性和审计能力。

容易扣分的回答：

- 把它说成完全自主 Agent。
- 不区分只读工具和写工具。
- 说“模型会自己判断是否退款”。

### Q16：你如何防止 LLM 编造订单、物流或退款结果？

考察点：幻觉治理。

我期望候选人这样回答：

> 项目用两层控制。第一层是业务事实从后端工具来，`get_order`、`get_logistics`、`get_product`、`apply_refund` 返回结构化上下文。第二层是 LLM prompt 明确要求只能基于后端工具结果回答：没有工具执行不能声称已退款、改地址或取消订单；`task_status` 处于 `awaiting_confirmation` 时只能请求确认。生产里还可以继续加输出校验，比如最终回答必须引用已有 `order_context` 字段，退款提交必须匹配工具返回的工单号。

容易扣分的回答：

- 只说“Prompt 写清楚不要编造”。
- 不提业务工具结果和后端 guardrail。

### Q17：真实生产中如何把 Mock 工具替换为业务系统？

考察点：工程落地能力。

我期望候选人这样回答：

> 我会把 `agent/tools.py` 中的 Mock 函数抽象成 adapter 接口，例如 `OrderClient`、`LogisticsClient`、`CatalogClient`、`RefundClient`。每个 adapter 要有统一输入输出 schema、timeout、retry、熔断、错误码映射和权限校验。退款这种写操作还要携带业务幂等键、审计信息和确认状态。测试里保留 mock provider，生产配置切到真实 provider，这样 Agent 图和 `/chat` 响应契约不需要大改。

容易扣分的回答：

- 只说“把 Mock API 地址换成真实接口”。
- 不提 timeout、错误映射、幂等和审计。

### Q18：你如何设计认证、租户隔离和 Admin 权限？

考察点：安全意识。

我期望候选人这样回答：

> 当前项目的缺口是 `user_id` 和 `session_id` 都来自请求体，Admin 接口也没有鉴权。生产里应该加认证依赖，比如 JWT、API key 或内部服务 token，从 token claims 派生 `tenant_id`、`user_id` 和 role，不信任请求体里的身份。普通用户只能访问自己的 session 和 memory，Admin 路由加 RBAC，记忆删除、会话查询、outbox 查询都写审计日志，并对响应做 PII 脱敏。

容易扣分的回答：

- 只说“加登录”。
- 不提不能信任请求体里的 `user_id`。
- 不提 Admin、审计和租户隔离。

### Q19：如果 RocketMQ 发送失败，会不会影响用户回复？

考察点：最终一致性和异步边界。

我期望候选人这样回答：

> 当前 `RocketMQPublisher.publish()` 会先写 outbox，再尝试发送 MQ。发送失败会把事件标记为 `enqueue_failed` 并记录 `last_error`，不会阻断 `/chat` 的用户响应。这是请求内 outbox 的教学实现。生产里应该让请求内只负责写 outbox，独立 relay worker 扫描 `pending/enqueue_failed`，按重试次数和退避策略发送，超过阈值进入 dead-letter，并提供 Admin 重放。

容易扣分的回答：

- 说“失败了再重试一下”但没有 outbox 状态机。
- 说“MQ 失败就让 `/chat` 失败”，没有区分用户可见主链路和后处理事件。

### Q20：为什么检索决策要支持 rules 和 LLM 两种模式？

考察点：稳定性和可演进性。

我期望候选人这样回答：

> rules 模式稳定、便宜、可测，适合订单、物流、退款、FAQ 这类强业务关键词场景。LLM 模式更灵活，可以做 query rewrite 和复杂意图判断，但必须输出结构化 JSON，而且失败时回退规则。这样既能保证当前系统可控，也给未来复杂意图识别留下演进空间。

容易扣分的回答：

- 认为 LLM 决策一定比规则好。
- 不提结构化输出和 fallback。

### Q21：如何评估长期记忆是否真的有效？

考察点：记忆系统评测能力。

我期望候选人这样回答：

> 不能只看“能保存”。要构造记忆评测集，包括偏好保存、偏好覆盖、投诉召回、删除后不召回、无关问题不召回、冲突记忆合并等场景。指标可以看 recall、precision、错误召回率、删除生效率和延迟。当前项目已经有关键词记忆、SQLite 语义记忆和 Milvus 接口，但 hash embedding 只是教学 fallback，生产要接真实 embedding，并记录 embedding model/version。

容易扣分的回答：

- 只说“问它记不记得”。
- 不提误召回、删除、生效延迟和评测集。

### Q22：这个项目的质量评估有什么局限？

考察点：是否能客观看待评测。

我期望候选人这样回答：

> 当前 `AutoQualityEvaluator` 是规则评分，按准确性、礼貌性、完整性加权。它适合做本地离线可测的质量信号和指标链路演示，但不能代表真实客服满意度。生产里应该结合人工标注、业务结果指标、转人工率、重复咨询率、投诉率，以及更严格的 LLM-as-judge 或 rubric 评估。

容易扣分的回答：

- 把当前 `quality_score` 当成绝对质量。
- 不提业务指标和人工标注。

### Q23：你会如何把 `/chat` 主路径异步化？

考察点：高并发 API 工程能力。

我期望候选人这样回答：

> 现在 `/chat` 是 async 入口，但内部还有同步 graph invoke、同步 SQLite/Postgres 连接、同步 outbox 和 RocketMQ 发送。改造方向是 LangGraph 用 `ainvoke`，节点里的工具调用尽量 async；Postgres 用 async 连接池，或同步代码统一放进受控线程池；RocketMQ 发送移出请求主路径，请求内只写 outbox；所有外部依赖都加 timeout、bulkhead 和降级策略。然后用 Locust 验证尾延迟和连接数。

容易扣分的回答：

- 只说“FastAPI 是异步的，所以没问题”。
- 不提同步 DB、同步 MQ 和连接池。

### Q24：如果 DeepSeek 配置缺失，为什么返回 503，而不是用规则草稿回答？

考察点：产品诚实性和运行模式边界。

我期望候选人这样回答：

> 当前产品定义是 `/chat` 必须走真实 LLM 主路径，规则回答只是后端草稿和工具上下文，不应该冒充模型最终能力。如果 DeepSeek key 缺失或 startup error 存在，就返回 `503 llm_unavailable`，并给出明确 reason。测试环境可以注入 fake LLM 保持稳定，但线上不能静默退回离线规则，否则用户和运营会误判系统能力。

容易扣分的回答：

- 说“为了省成本可以随时用规则兜底”。
- 不区分测试 fake LLM 和线上真实 LLM。

### Q25：`request_id` 幂等和 `apply_refund` 幂等有什么区别？

考察点：多层幂等设计。

我期望候选人这样回答：

> `request_id` 是 API 层逻辑请求幂等，防止客户端重试导致同一条消息重复处理。`apply_refund` 的 `idempotency_key` 是业务写操作幂等，防止同一退款动作重复创建工单。两者作用域不同：一个保护 `/chat` 请求，一个保护业务工具副作用。即使 API 层因为重试或恢复异常重复走到工具层，写工具也应该有自己的业务幂等保护。

容易扣分的回答：

- 认为有 `request_id` 就不需要退款幂等。
- 认为退款幂等只在 Mock 里有意义。

### Q26：你会如何用测试证明退款不会越权执行？

考察点：测试意识。

我期望候选人这样回答：

> 我会覆盖四类测试：用户只说退款但没订单号时，只返回补槽不执行工具；有订单号但未确认时，停在 `awaiting_confirmation` 并返回 choices；用户确认后，`tool_executor` 调用 `apply_refund` 且携带 `idempotency_key`；重复确认或重复请求不会创建第二个工单。当前 `test_graph_skeleton.py` 和 `test_chat_flow.py` 已经覆盖了补槽、确认门、写工具幂等键和多轮退款流程。

容易扣分的回答：

- 只说“手工点一下看结果”。
- 不区分图节点测试和 API 端到端测试。

### Q27：Prometheus、LangSmith、日志分别解决什么问题？

考察点：可观测性分层。

我期望候选人这样回答：

> Prometheus 看聚合指标，比如请求数、延迟、Token、质量分、错误数和转人工数；LangSmith metadata 用于跟踪单次 LangGraph/LLM 调用，带上 `session_id`、`user_id`、环境和版本；日志用于异常排查和审计，例如质量告警、LLM 不可用、outbox 发送失败。三者不是替代关系，分别面向运营指标、链路追踪和事件排障。

容易扣分的回答：

- 说“接了监控”但讲不出指标。
- 把 `/metrics` 和单次 trace 混为一谈。

### Q28：如果让你只选一个点继续优化，你会先做什么？

考察点：优先级判断。

我期望候选人这样回答：

> 如果目标是面向真实用户，我会先做认证授权和租户隔离，因为当前用户身份来自请求体，Admin 接口也没有保护，这是上线前 P0。随后做主路径异步化和独立 outbox relay，因为它们影响并发吞吐和事件可靠性。真实业务工具、embedding 和评测增强也重要，但要排在安全边界和主链路稳定性之后。

容易扣分的回答：

- 先说“换更强模型”。
- 只关注模型效果，不关注身份、数据和稳定性。

---

## 20. 面试官评分标准

面试官通常不是在检查你能不能背出“FastAPI、LangGraph、Redis、Postgres”，而是在判断你是否真的具备大模型应用工程能力。

### 合格候选人

应该能讲清：

- `/chat` 的真实调用链。
- Agent 图为什么要有状态机、补槽和确认门。
- 只读工具和写工具的权限边界。
- 短期记忆、摘要记忆、长期记忆、语义记忆分别解决什么问题。
- 幂等、会话锁、乐观锁分别防什么风险。
- LLM 不可用、RAG 不可用、Token 超限时怎么降级。
- 当前项目哪些地方是教学实现，哪些地方离生产还有差距。

### 优秀候选人

除了能讲清现状，还会主动补充：

- API 层不能信任请求体里的 `user_id`，要从认证上下文派生身份。
- `SessionStore` 和 LangGraph checkpointer 的职责不同。
- 写工具必须有业务幂等、审计和权限，不只是 prompt 约束。
- Outbox 当前没有 relay worker，不能宣称完整最终一致性闭环。
- 语义记忆当前 hash embedding 不是生产级召回。
- FastAPI async 入口里同步 DB 和同步 MQ 会阻塞事件循环。
- 质量评分是规则信号，不是最终用户满意度。
- 自动评测要覆盖多轮、工具失败、删除记忆、低置信度和人工接管。

### 明显扣分回答

这些回答会让面试官怀疑你没有真正做过生产级 Agent：

- “这个项目就是调用大模型回答客服问题。”
- “LangGraph 会自动处理所有状态。”
- “只要 prompt 写好，模型不会乱退款。”
- “用了 Redis 就不会并发冲突。”
- “有 RAG 就不会幻觉。”
- “接了 Prometheus 就算可观测了。”
- “有 Docker Compose 就是生产部署。”
- “质量分 90 就说明用户满意。”
- “后续主要换一个更强模型。”

---

## 21. 生产就绪差距

当前已完成：

- FastAPI `/chat` 和内置客服 UI。
- LangGraph 多轮状态机。
- 只读 ReAct 三步上限。
- 退款确认门和写工具幂等键。
- SQLite/Postgres 会话和用户记忆。
- 摘要记忆和语义记忆接口。
- request_id 幂等。
- Redis/内存会话锁。
- 用户限流、全局 QPS、Token 预算降级。
- DeepSeek 真实 LLM 主路径。
- LLM fallback 和熔断测试层。
- FAQ/RAG 适配。
- Prometheus 指标、LangSmith metadata、质量评估。
- RocketMQ outbox 和事件 envelope。
- Docker Compose、Grafana、Locust、自动评测。

仍需补齐：

- API 认证、RBAC、租户隔离、Admin 保护。
- PII 脱敏和敏感操作审计。
- `/chat` 主路径里的同步图调用、同步 DB、同步 MQ 发送改造。
- Postgres async 连接池或受控线程池。
- 独立 outbox relay、死信队列、重放接口。
- 真实订单、物流、商品、退款 adapter。
- 真实 embedding 服务和 Milvus 召回质量验证。
- 记忆抽取、冲突合并、过期和用户可编辑。
- 真实依赖健康检查。
- 24 小时长稳运行和正式压测报告。

---

## 22. 复习清单

能讲清以下内容，就基本掌握了 05 项目：

- `POST /chat` 的完整调用链。
- `request_id` 幂等如何防重复请求。
- Redis 会话锁和 Session version 乐观锁分别解决什么问题。
- LangGraph 图中每个节点的职责。
- `dialog_state` 如何支撑多轮补槽、确认和过期。
- 为什么退款是写工具，必须经过确认门。
- 只读 ReAct 和完整 autonomous agent 的区别。
- 三层记忆分别存什么，按什么 key 隔离。
- 检索决策如何控制 memory / FAQ / business context。
- 真实 LLM 如何被强制作为主路径。
- Prometheus、LangSmith metadata、质量评估分别解决什么观测问题。
- RocketMQ outbox 当前做到哪一步，缺什么。
- Docker Compose 启动了哪些依赖。
- pytest、evals、Locust 分别验证什么。
- 面试中如何主动讲清已完成能力和生产差距。

---

## 23. 最短面试版讲解

如果面试官只给你 2 分钟，可以这样讲：

> 05 项目是一个生产级智能客服 Agent。用户通过 FastAPI `/chat` 发起请求，请求必须携带 `request_id`，服务端用幂等表防止客户端重试导致重复处理，再用 Redis 会话锁和数据库 version 防止同一会话并发覆盖。进入 Agent 前会加载会话、摘要记忆、长期记忆和语义记忆，并通过检索决策判断是否需要 FAQ/RAG 或业务上下文。核心 Agent 是 LangGraph 多轮状态机，不是完全开放的自主 Agent；它通过 `dialog_state` 管理补槽、确认、过期和任务阶段，只读问题可以进入最多 3 步 ReAct 工具循环，退款这种写操作必须经过确认门并携带业务幂等键。工具结果进入 `response_builder` 形成规则草稿，再由 DeepSeek 真实 LLM 基于后端上下文生成最终客服话术，避免编造订单和退款事实。最后系统保存会话、提取长期记忆、记录 Prometheus 指标、做质量评估，并通过 RocketMQ outbox 发布对话完成和后处理事件。项目还提供 Docker Compose、Grafana、Locust 和 100 题自动评测。它已经具备生产工程化骨架，但真实上线还要补认证授权、异步主路径、真实业务工具、独立 outbox relay、真实 embedding 和长稳压测。
