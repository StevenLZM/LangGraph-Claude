# 工程设计：生产级 AI Agent 平台 — 智能客服系统

> 对应 PRD：05_production_agent_customer_service.md

---

## 一、整体架构图

```
                           互联网用户
                               │
                    ┌──────────▼──────────┐
                    │     Nginx / CDN      │
                    │   (限流 + SSL终止)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   FastAPI 服务集群   │
                    │  (uvicorn + async)  │
                    └──┬───────────────┬──┘
                       │               │
          ┌────────────▼───┐   ┌───────▼────────────┐
          │  Redis 缓存层   │   │   消息队列 (可选)   │
          │ - 限流计数器    │   │   Celery/ARQ       │
          │ - 会话分布式锁  │   │   RocketMQ 事件     │
          └────────────┬───┘   └───────────────────┘
                       │
          ┌────────────▼──────────────────────────────┐
          │           LangGraph Agent 层               │
          │                                            │
          │  ┌────────┐  ┌────────┐  ┌─────────────┐ │
          │  │会话上下文│ │摘要/长期│ │   工具调用   │ │
          │  │Token管理│  │Milvus可选│ │(订单/物流)  │ │
          │  └────────┘  └────────┘  └─────────────┘ │
          └──────────────────┬────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │              LLM 调用层                     │
          │  DeepSeek / OpenAI-compatible / Anthropic   │
          │  (主备模型 + 重试 + 指数退避 + 熔断)        │
          └──────────────────┬────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │            可观测性层                       │
          │  LangSmith(链路) + Prometheus(指标)        │
          │  Grafana(仪表盘) + AlertManager(告警)      │
          └───────────────────────────────────────────┘
          
          ┌──────────────────────────────────────────┐
          │          基础设施层                        │
          │  PostgreSQL(会话/记忆/幂等/outbox)          │
          │  Redis(限流/锁) + RocketMQ + Compose       │
          └──────────────────────────────────────────┘
```

---

## 二、生产并发与幂等设计

多实例部署时，`/chat` 不是普通读接口：一次请求会调用 LLM、保存会话、提取长期记忆、记录指标并发布业务消息。客户端超时后重试、浏览器重复点击、网关重放、同一会话并发发送，都可能导致重复扣费、重复消息、会话覆盖或顺序错乱。因此 API 边界必须把“同一条消息只处理一次”和“同一会话有序写入”作为基础能力。

### 2.1 客户端协议：稳定 `request_id`

`POST /chat` 请求包含四个必填字段：

```json
{
  "user_id": "user_001",
  "session_id": "session_001",
  "request_id": "req_001",
  "message": "我的订单 ORD123456 到哪了？"
}
```

约定：

- 客户端每发送一条新消息生成一个新的 `request_id`，浏览器 UI 使用 `crypto.randomUUID()`。
- 同一条消息重试时必须复用同一个 `request_id`。
- 幂等作用域是 `(user_id, session_id, request_id)`，避免不同用户或不同会话之间误判重复。
- 服务端额外计算 `message_hash = sha256(user_id + session_id + message)`；同一个 `request_id` 如果换了消息内容，返回 `409 idempotency_key_conflict`。

响应中会回传：

- `request_id`：便于客户端和日志对齐。
- `request_status`：`processed` 表示首次处理，`replayed` 表示重复请求返回了首次结果，`degraded` 表示 Token 预算降级。

### 2.2 服务端幂等表：Postgres 为生产主路径

幂等状态由 `api/idempotency.py` 管理，本地测试用 SQLite，生产 `STORAGE_BACKEND=postgres` 时使用 Postgres：

```text
chat_requests
  user_id
  session_id
  request_id
  message_hash
  status              processing / succeeded / failed
  response_json       首次成功响应快照
  error_status_code
  error_detail_json
  created_at
  updated_at
  PRIMARY KEY(user_id, session_id, request_id)
```

处理流程：

```text
1. start_request() 尝试插入 processing
2. 插入成功：当前实例拥有该逻辑请求，继续处理
3. 插入失败：说明是重复请求，读取已有记录
4. message_hash 不一致：409 idempotency_key_conflict
5. 已 succeeded：反序列化 response_json，返回 request_status=replayed
6. 已 failed：按首次失败的 status/detail 返回
7. 仍 processing：最多等待 1.5s；若仍未完成，返回 202 request_processing
```

这个设计解决客户端重试同一条消息的问题：客户端可以安全重试，不会重复调用 LLM，也不会重复发布客服完成事件。`GET /chat/requests/{request_id}?user_id=...&session_id=...` 用于查询请求状态，适合客户端收到 `202` 后轮询或运维排障。

### 2.3 同会话并发：Redis 分布式锁 + 会话乐观锁

幂等解决“同一条消息重复处理”，但不能解决“同一会话两条不同消息同时处理”。例如用户连续发送 A、B，两台 API 实例同时加载同一个旧会话，如果都写回，后写入可能覆盖先写入。

当前使用两层保护：

```text
SessionLockManager
  Redis: SET lock:chat_session:{session_id} <token> NX PX <lease>
  release: Lua 校验 token 后删除
  local tests: asyncio.Lock fallback

SessionStore.save_session(expected_version=N)
  load_session() 返回 version
  保存时 WHERE session_id=? AND version=?
  stale version 返回 SessionVersionConflict -> 409
```

Redis 锁让同一个 `session_id` 的新请求串行进入核心处理，减少 LLM 和工具链路的竞态。乐观锁是最后一道数据库一致性保护：即便锁租约过期、实例重启或未来有旁路写入，也不会静默覆盖会话。

### 2.4 业务消息一致性：Postgres Outbox

`/chat` 成功生成用户可见回答后，会发布 RocketMQ 事件：

- `ChatCompleted`
- `PostprocessRequested`
- `HumanTransferReminderRequested`

生产模式下 outbox 也落到 Postgres，避免 API 多实例各自写本地 SQLite 文件。流程是：

```text
1. 构造 RocketMQ event，event_id 全局唯一
2. 写入 message_outbox，event_id 为主键，重复写入忽略
3. 发送 RocketMQ
4. 成功后标记 published；失败后标记 enqueue_failed 并记录 last_error
5. /admin/messages/outbox 可查看 outbox 状态
```

当前发布器仍是请求内发送，因此它是“本地 outbox + 可观测失败”的实现，不是完整独立 relay worker。生产进一步增强时，应把 outbox pending/enqueue_failed 事件交给独立 worker 重试，并用 `FOR UPDATE SKIP LOCKED` 控制多 worker 并发。

### 2.5 为什么选 Redis + Postgres

生产选型：

- Postgres：承担幂等记录、会话状态、用户长期记忆、RocketMQ outbox 和 LangGraph Postgres checkpoint。它提供唯一约束、事务更新、乐观锁和可查询审计记录。
- Redis：承担高频、短 TTL、跨实例共享的限流计数和会话分布式锁。它适合 `INCR/EXPIRE`、`SET NX PX` 这类低延迟协调，不适合保存长期业务事实。
- SQLite / in-memory：只用于本地演示和 pytest，不作为多实例生产选型。

核心原则是：业务事实放 Postgres，短期协调放 Redis，跨系统通知走 RocketMQ，客户端重试通过 `request_id` 明确语义。

---

## 三、LangGraph Agent 设计

### 3.1 状态定义

```python
# agent/state.py
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages

class CustomerServiceState(TypedDict):
    # 会话标识
    session_id: str
    user_id: str
    
    # 消息管理
    messages: Annotated[list, add_messages]  # 当前窗口内消息
    window_size: int                          # 当前窗口消息数
    total_turns: int                          # 总对话轮次
    
    # 用户上下文
    user_profile: dict                        # 用户基本信息
    user_memories: List[str]                  # 从长期记忆召回的内容
    memory_summary: str                       # 从摘要记忆加载的会话摘要
    retrieval_decision: dict                  # 检索决策：是否查记忆、FAQ、业务上下文
    order_context: Optional[dict]             # 当前讨论的订单信息
    
    # 执行控制
    needs_human_transfer: bool                # 是否需要转人工
    transfer_reason: str                      # 转人工原因
    
    # 监控
    token_used: int                           # 本次调用 token 数
    response_time_ms: int                     # 响应时间
    quality_score: Optional[int]              # 自动评估分数
```

### 3.2 Graph 结构

```python
# agent/graph.py
from langgraph.graph import START, END, StateGraph

def build_customer_service_graph():
    workflow = StateGraph(CustomerServiceState)

    workflow.add_node("context_loader", context_loader_node)
    workflow.add_node("turn_router", turn_router_node)
    workflow.add_node("pending_task_resolver", pending_task_resolver_node)
    workflow.add_node("retrieval_decision", retrieval_decision_node)
    workflow.add_node("slot_filling", slot_filling_node)
    workflow.add_node("confirmation_guard", confirmation_guard_node)
    workflow.add_node("tool_planner_or_react", tool_planner_or_react_node)
    workflow.add_node("tool_executor", tool_executor_node)
    workflow.add_node("response_builder", response_builder_node)
    workflow.add_node("finalizer", finalizer_node)

    workflow.add_edge(START, "context_loader")
    workflow.add_edge("context_loader", "turn_router")
    workflow.add_conditional_edges("turn_router", route_after_turn_router, ...)
    workflow.add_edge("pending_task_resolver", "slot_filling")
    workflow.add_edge("retrieval_decision", "slot_filling")
    workflow.add_conditional_edges("slot_filling", route_after_slot_filling, ...)
    workflow.add_conditional_edges("confirmation_guard", route_after_confirmation_guard, ...)
    workflow.add_edge("tool_planner_or_react", "tool_executor")
    workflow.add_conditional_edges("tool_executor", route_after_tool_executor, ...)
    workflow.add_edge("response_builder", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)
```

说明：当前生产边界仍由 FastAPI 主链路负责加载会话、摘要记忆、检索决策、长期记忆、限流和持久化；LangGraph 图负责多轮任务状态推进。`dialog_state` 保存当前任务、阶段、槽位、确认状态、过期时间和幂等键；缺槽时停在 `slot_filling -> response_builder`，写操作必须经过 `confirmation_guard`，只读复杂问题可在 `tool_planner_or_react -> tool_executor` 内最多循环 3 步。这样既保留 `/chat` 的幂等、会话锁、版本保存和消息 outbox 边界，也让十轮以上连续对话不依赖自然语言历史猜状态。

生产级收口：

- 只读 ReAct 的典型路径是“退款资格 + 物流追问”：`get_order -> get_logistics -> faq_rag`。三步均为只读，超过 `max_tool_steps=3` 后必须停止循环。
- `response_builder` 会合并同轮多工具结果。物流上下文和 FAQ/RAG 的 `rag_matched`、`rag_sources`、`rag_backend` 进入同一个 `order_context`，避免最终回答只呈现最后一个工具结果。
- 写工具不能依赖 LLM 决策。`apply_refund` 只能在 `confirmation_guard` 放行后由 `tool_executor` 调用，并且调用参数必须包含 `dialog_state.idempotency_key`。
- Mock 售后工具维护退款提交账本，同一 `idempotency_key` 重放返回同一工单；真实生产系统应把该账本迁移到售后系统或业务数据库事务中。

---

## 四、记忆系统设计

### 4.1 短期记忆：Token 窗口管理

```python
# memory/short_term.py
from langchain_core.messages import trim_messages, SystemMessage, AIMessage, HumanMessage

class ContextWindowManager:
    """
    管理对话 Token 窗口，防止超出模型限制
    策略：保留最新消息，压缩/摘要早期消息
    """
    
    MAX_TOKENS = 3500          # 总 Token 上限（给 system prompt 留 500）
    SUMMARY_THRESHOLD = 2500   # 超过此值触发压缩
    
    def __init__(self, llm):
        self.llm = llm
        self.summary_llm = get_small_llm()  # 用小模型做摘要，省成本
    
    async def trim(self, messages: list) -> list:
        """裁剪消息到窗口大小"""
        # 计算当前 token 数
        current_tokens = await self._count_tokens(messages)
        
        if current_tokens <= self.SUMMARY_THRESHOLD:
            return messages  # 不需要裁剪
        
        # 找分割点：保留最近 8 轮，压缩更早的
        recent_messages = messages[-16:]  # 最近 8 轮（每轮=human+ai）
        old_messages = messages[:-16]
        
        if not old_messages:
            # 仍然太长，强制截断
            return trim_messages(
                messages,
                max_tokens=self.MAX_TOKENS,
                strategy="last",
                token_counter=self.llm,
                include_system=True
            )
        
        # 摘要早期对话
        summary = await self._summarize(old_messages)
        summary_msg = SystemMessage(
            content=f"[早期对话摘要] {summary}",
            additional_kwargs={"type": "summary"}
        )
        
        return [summary_msg] + recent_messages
    
    async def _summarize(self, messages: list) -> str:
        """将早期消息摘要为一段文字"""
        msg_text = "\n".join([
            f"{'用户' if isinstance(m, HumanMessage) else '客服'}: {m.content}"
            for m in messages if hasattr(m, "content")
        ])
        
        result = await self.summary_llm.ainvoke(
            f"请用100字以内概括以下对话的关键信息（用户问题、已解决的问题等）：\n{msg_text}"
        )
        return result.content
    
    async def _count_tokens(self, messages: list) -> int:
        """计算消息列表的 token 数"""
        # 使用模型的 get_num_tokens_from_messages 方法
        return self.llm.get_num_tokens_from_messages(messages)


# 节点实现
async def context_trimmer_node(state: CustomerServiceState) -> dict:
    manager = ContextWindowManager(llm)
    trimmed_messages = await manager.trim(state["messages"])
    
    return {
        "messages": trimmed_messages,
        "window_size": len(trimmed_messages)
    }
```

### 4.2 摘要记忆：会话级滚动摘要

摘要记忆独立于短期窗口和长期用户画像，按 `session_id` 保存。它解决的问题是：短期窗口只能保留最近消息，但长会话重启或后续追问仍需要早期背景。

```python
# memory/summary.py
class SummaryMemoryStore:
    def load_summary(self, session_id: str) -> SummaryMemoryRecord | None: ...

    def save_summary(
        self,
        *,
        session_id: str,
        user_id: str,
        summary: str,
        covered_turns: int,
        source_event_id: str = "",
    ) -> SummaryMemoryRecord: ...
```

主链路读取摘要：

```python
loaded_summary = summary_memory_store.load_summary(req.session_id)
summary_message = _summary_system_message(loaded_summary.summary if loaded_summary else "")
messages = context_window_manager.trim([*summary_message, *prior_messages, HumanMessage(content=req.message)])
```

异步写入摘要：

```python
summary_record = summary_memory_store.save_summary(
    session_id=session_id,
    user_id=user_id,
    summary=build_conversation_summary(messages),
    covered_turns=sum(1 for message in messages if isinstance(message, HumanMessage)),
    source_event_id=event.event_id,
)
```

### 4.3 长期记忆：关键词/结构化 + 语义检索

长期记忆按 `user_id` 隔离，服务跨会话用户画像。当前实现分两层：

- `UserMemoryManager`：SQLite/Postgres 关键词和结构化记忆，保存偏好、投诉、用户资料。
- `SemanticMemoryStore`：语义长期记忆接口，默认 SQLite fallback，可配置 Milvus Lite backend。

```python
# memory/semantic.py
class StructuredMemory:
    user_id: str
    category: str
    entity_type: str
    key: str
    value: dict
    content: str
    source_session_id: str
    source_request_id: str

class SemanticMemoryStore(Protocol):
    def search(self, *, user_id: str, query: str, filters: dict | None = None, limit: int = 5): ...
    def upsert_from_turn(self, *, user_id: str, user_message: str, assistant_answer: str, session_id: str, request_id: str): ...
    def delete_user(self, user_id: str) -> int: ...
```

Milvus backend 配置：

```bash
SEMANTIC_MEMORY_BACKEND=milvus
MILVUS_MEMORY_URI=./data/memory_milvus.db
MILVUS_MEMORY_COLLECTION=customer_long_term_memories
```

当前 Milvus backend 使用确定性 hash embedding 保持离线可测。生产可替换为真实 embedding 服务，但不应改变 `/chat` 响应契约和 `SemanticMemoryStore` 接口。

### 4.4 检索决策节点

检索决策节点决定是否读取长期记忆、FAQ/RAG 或业务上下文。默认规则决策，`RETRIEVAL_DECISION_MODE=llm` 时可使用真实 LLM 输出 JSON；解析失败时退回规则决策。

```python
RetrievalDecision(
    intent="logistics",
    needs_memory=True,
    needs_faq=False,
    needs_business_context=False,
    memory_filters={"category": ["delivery_preference", "preference"]},
    external_sources=["long_term_memory"],
    query_rewrite="给我送货用什么快递？",
)
```

检索决策不负责业务安全判断。退款二次确认、投诉/法律转人工等 guardrail 仍由 `turn_router`、`slot_filling` 和 `confirmation_guard` 等确定性图节点负责。

---

## 五、限流设计

```python
# middleware/rate_limiter.py
import redis.asyncio as aioredis
from fastapi import Request, HTTPException

class RateLimiter:
    """
    多层限流：
    - 用户级：每分钟 10 次请求
    - 全局 Token：每小时预算
    - QPS：全局每秒请求数
    """
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def check_user_rate(self, user_id: str) -> None:
        """用户级限流：令牌桶算法"""
        key = f"ratelimit:user:{user_id}:{int(time.time() // 60)}"
        
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 120)  # 2分钟TTL（容错）
        results = await pipe.execute()
        
        count = results[0]
        if count > 10:
            # 计算剩余等待时间
            ttl = await self.redis.ttl(key)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": "请求过于频繁，请稍后再试",
                    "retry_after": ttl
                }
            )
    
    async def check_token_budget(self, tokens_to_use: int = 1000) -> None:
        """全局 Token 预算控制"""
        hour_key = f"token_budget:{int(time.time() // 3600)}"
        
        current = int(await self.redis.get(hour_key) or 0)
        HOURLY_BUDGET = 500_000  # 每小时 50万 tokens 预算
        
        if current + tokens_to_use > HOURLY_BUDGET:
            # 降级：切换到更便宜的模型或简化回答
            raise TokenBudgetExceeded("Token预算已满，降级处理")
        
        await self.redis.incrby(hour_key, tokens_to_use)
        await self.redis.expire(hour_key, 7200)
    
    async def check_global_qps(self) -> None:
        """全局 QPS 控制"""
        key = f"qps:{int(time.time())}"
        count = await self.redis.incr(key)
        await self.redis.expire(key, 2)
        
        if count > 100:  # 全局每秒 100 QPS
            raise HTTPException(status_code=503, detail="服务繁忙，请稍后重试")
```

---

## 六、LLM 降级与重试

```python
# llm/resilient_llm.py
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class ResilientLLM:
    """
    带弹性的 LLM 调用：
    - 自动重试（指数退避）
    - 主备模型切换
    - 熔断器（连续失败后暂停）
    """
    
    def __init__(self):
        self.primary = ChatAnthropic(model="claude-sonnet-4-6", max_retries=0)
        self.fallback = ChatOpenAI(model="gpt-4o-mini", max_retries=0)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_time=60)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((APIError, RateLimitError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def invoke(self, messages: list, use_fallback: bool = False) -> str:
        llm = self.fallback if use_fallback else self.primary
        
        try:
            if self.circuit_breaker.is_open():
                # 熔断器开启，直接用备用模型
                return await self.fallback.ainvoke(messages)
            
            response = await llm.ainvoke(messages)
            self.circuit_breaker.record_success()
            return response
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            if not use_fallback:
                logger.warning(f"主模型失败，切换备用: {e}")
                return await self.invoke(messages, use_fallback=True)
            raise


class CircuitBreaker:
    """简单熔断器实现"""
    
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.last_failure_time = None
        self.state = "closed"  # closed/open/half-open
    
    def is_open(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = "half-open"
                return False
            return True
        return False
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"熔断器开启！连续失败 {self.failure_count} 次")
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
```

---

## 七、监控与可观测性

### 7.1 LangSmith 链路追踪

```python
# monitoring/tracing.py
import os
from langsmith import Client
from langchain_core.callbacks import LangChainTracer

def setup_langsmith():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "production-customer-service"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

def get_session_tracer(session_id: str, user_id: str) -> LangChainTracer:
    return LangChainTracer(
        tags=[f"session:{session_id}", f"user:{user_id}"],
        metadata={
            "environment": os.getenv("ENV", "production"),
            "version": "1.0.0"
        }
    )
```

### 7.2 Prometheus 指标

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
REQUEST_COUNT = Counter(
    "agent_requests_total",
    "总请求数",
    ["status", "user_type"]  # status: success/error/transferred
)

# 响应时间
RESPONSE_TIME = Histogram(
    "agent_response_time_seconds",
    "响应时间分布",
    buckets=[0.5, 1, 2, 5, 10, 30]
)

# Token 消耗
TOKEN_USAGE = Counter(
    "agent_tokens_total",
    "Token 总消耗",
    ["model", "type"]  # type: input/output
)

# 活跃会话数
ACTIVE_SESSIONS = Gauge("agent_active_sessions", "当前活跃会话数")

# 质量评分
QUALITY_SCORE = Histogram(
    "agent_quality_score",
    "回答质量评分分布",
    buckets=[0, 20, 40, 60, 70, 80, 90, 100]
)

# 在 FastAPI 中暴露指标
from prometheus_client import make_asgi_app
metrics_app = make_asgi_app()
```

### 7.3 自动质量评估

```python
# monitoring/evaluator.py
from langchain.evaluation import load_evaluator

class AutoQualityEvaluator:
    """使用 LLM 自动评估回答质量，低分触发告警"""
    
    PASS_SCORE = 70
    
    def __init__(self):
        # 用更便宜的模型做评估，节省成本
        self.eval_llm = get_small_llm()
    
    async def evaluate(self, question: str, answer: str, context: dict) -> dict:
        eval_prompt = f"""
评估以下客服对话的质量（每项0-100分）：

用户问题：{question}
客服回答：{answer}
用户信息：{json.dumps(context, ensure_ascii=False)}

评估标准：
1. 准确性（40%）：回答是否正确解决了用户问题
2. 礼貌性（30%）：语气是否专业友好
3. 完整性（30%）：是否完整回答，有没有遗漏关键信息

请输出JSON：
{{"accuracy": 整数, "politeness": 整数, "completeness": 整数,
  "score": 加权总分, "passed": true/false, "issues": ["问题1"]}}
"""
        result = await self.eval_llm.ainvoke(eval_prompt)
        eval_data = json.loads(result.content)
        
        # 记录到 Prometheus
        QUALITY_SCORE.observe(eval_data["score"])
        
        # 低分告警
        if not eval_data["passed"]:
            await self.trigger_alert(question, answer, eval_data)
        
        return eval_data
    
    async def trigger_alert(self, question, answer, eval_data):
        logger.warning(
            f"质量告警 score={eval_data['score']} "
            f"issues={eval_data['issues']}"
        )
        # 发送到告警系统（Slack/钉钉/PagerDuty）
        await send_alert({
            "level": "warning",
            "message": f"客服回答质量低（{eval_data['score']}分）",
            "details": eval_data
        })
```

---

## 八、MCP 集成设计

### 8.1 业务工具 MCP Server

```python
# mcp_servers/order_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("order-service")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_order",
            description="查询订单详情，包括商品、金额、状态、物流信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "user_id": {"type": "string"}
                },
                "required": ["order_id"]
            }
        ),
        Tool(
            name="query_logistics",
            description="查询物流实时状态和预计送达时间",
            inputSchema={
                "type": "object",
                "properties": {
                    "tracking_number": {"type": "string"},
                    "carrier": {"type": "string"}
                },
                "required": ["tracking_number"]
            }
        ),
        Tool(
            name="initiate_refund",
            description="发起退款申请（仅在用户明确要求且符合退款条件时使用）",
            inputSchema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "amount": {"type": "number"}
                },
                "required": ["order_id", "reason"]
            }
        ),
        Tool(
            name="query_product",
            description="查询商品详情、库存、价格",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "sku": {"type": "string"}
                }
            }
        )
    ]
```

### 8.2 MCP 配置

```json
{
  "mcpServers": {
    "order-service": {
      "command": "python",
      "args": ["-m", "mcp_servers.order_server"],
      "env": {
        "ORDER_API_BASE": "${ORDER_API_BASE}",
        "ORDER_API_KEY": "${ORDER_API_KEY}"
      }
    },
    "knowledge-base": {
      "command": "python",
      "args": ["-m", "mcp_servers.kb_server"],
      "description": "FAQ知识库查询（基于项目01的RAG系统）"
    },
    "crm": {
      "command": "python",
      "args": ["-m", "mcp_servers.crm_server"],
      "description": "CRM系统：查询/更新用户信息"
    }
  }
}
```

---

## 九、System Prompt 工程设计

```python
# agent/prompts.py

CUSTOMER_SERVICE_PROMPT = """你是"小智"，{company_name} 的 AI 客服助手。

【用户信息】
姓名：{user_name}
会员等级：{membership_level}
历史背景：{user_memories}

【你的能力】
- 查询订单状态、物流信息
- 解答商品咨询、退换货政策
- 处理账户问题
- 发起退款申请（需用户明确同意）

【行为准则】
1. 称呼用户为"{user_name}"，保持亲切专业
2. 回答简洁，不超过200字，复杂问题分步骤说明
3. 涉及金额操作（退款等），必须向用户确认后再执行
4. 遇到以下情况立即转人工：
   - 用户明确要求转人工
   - 涉及投诉/纠纷/法律问题
   - 超过2次无法解决用户问题
   - 情绪激动的用户

【禁止行为】
- 不承诺公司政策之外的补偿
- 不透露系统内部信息
- 不对无法确认的事实做出保证

当前日期：{current_date}
"""

def build_system_prompt(user_id: str, user_memories: List[str]) -> str:
    user_info = get_user_info(user_id)
    return CUSTOMER_SERVICE_PROMPT.format(
        company_name="XX电商",
        user_name=user_info.get("name", "亲"),
        membership_level=user_info.get("level", "普通会员"),
        user_memories="\n".join(f"- {m}" for m in user_memories) or "暂无历史记录",
        current_date=datetime.now().strftime("%Y年%m月%d日")
    )
```

---

## 十、Docker Compose 部署

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - PG_HOST=postgres
      - LANGCHAIN_API_KEY=${LANGSMITH_API_KEY}
      - LANGCHAIN_TRACING_V2=true
    depends_on:
      - redis
      - postgres
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: customer_memories
      POSTGRES_PASSWORD: ${PG_PASSWORD}
    volumes:
      - pg_data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus
    volumes:
      - ./infra/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./infra/grafana:/etc/grafana/provisioning

volumes:
  redis_data:
  pg_data:
```

---

## 十一、目录结构

```
05_production_customer_service/
├── api/
│   ├── main.py                  # FastAPI 入口 + 指标端点
│   ├── routers/
│   │   ├── chat.py              # 对话接口
│   │   └── admin.py             # 管理接口
│   └── middleware/
│       ├── rate_limiter.py      # 限流中间件
│       └── auth.py              # 认证中间件
├── agent/
│   ├── graph.py                 # LangGraph 图
│   ├── state.py                 # 状态定义
│   ├── nodes.py                 # 所有节点实现
│   ├── retrieval.py             # 检索决策
│   ├── tools.py                 # 工具定义
│   └── prompts.py               # Prompt 模板
├── memory/
│   ├── short_term.py            # Token 窗口管理
│   ├── summary.py               # 会话摘要记忆
│   ├── semantic.py              # SQLite/Milvus 语义长期记忆
│   └── long_term.py             # 用户关键词/结构化长期记忆
├── llm/
│   └── resilient_llm.py         # 弹性 LLM（重试+熔断）
├── mcp_servers/
│   ├── order_server.py          # 订单 MCP Server
│   ├── kb_server.py             # 知识库 MCP Server
│   └── crm_server.py            # CRM MCP Server
├── monitoring/
│   ├── tracing.py               # LangSmith 配置
│   ├── metrics.py               # Prometheus 指标
│   └── evaluator.py             # 自动质量评估
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana/                 # Grafana 仪表盘 JSON
├── tests/
│   ├── test_rate_limiter.py
│   ├── test_memory.py
│   ├── test_llm_fallback.py
│   └── load_test.py             # locust 压力测试
├── .mcp.json
├── .env.example
└── requirements.txt
```

---

## 十二、性能基准测试

```python
# tests/load_test.py - 使用 locust
import time

from locust import HttpUser, task, between

class CustomerServiceUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def ask_order_status(self):
        self.client.post("/chat", json={
            "user_id": "test_user_001",
            "session_id": "test_user_001_session_order",
            "request_id": f"test_user_001_order_{time.time_ns()}",
            "message": "我的订单ORD123456到哪了？"
        })
    
    @task(2)
    def ask_product_info(self):
        self.client.post("/chat", json={
            "user_id": "test_user_002",
            "session_id": "test_user_002_session_product",
            "request_id": f"test_user_002_product_{time.time_ns()}",
            "message": "这个商品支持7天无理由退货吗？"
        })
    
    @task(1)
    def request_refund(self):
        self.client.post("/chat", json={
            "user_id": "test_user_003",
            "session_id": "test_user_003_session_refund",
            "request_id": f"test_user_003_refund_{time.time_ns()}",
            "message": "我要申请退款，商品有质量问题"
        })

# 运行命令: locust -f tests/load_test.py --host=http://localhost:8000
# 目标: 50并发用户，平均响应时间 < 5s，错误率 < 1%
```
