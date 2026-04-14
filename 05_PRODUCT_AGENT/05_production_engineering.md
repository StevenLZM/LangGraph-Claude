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
          │ - 热点会话缓存  │   │   (异步任务)        │
          └────────────┬───┘   └───────────────────┘
                       │
          ┌────────────▼──────────────────────────────┐
          │           LangGraph Agent 层               │
          │                                            │
          │  ┌────────┐  ┌────────┐  ┌─────────────┐ │
          │  │短期记忆 │  │长期记忆 │  │   工具调用   │ │
          │  │Token管理│  │Mem0/PG │  │(订单/物流)  │ │
          │  └────────┘  └────────┘  └─────────────┘ │
          └──────────────────┬────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │              LLM 调用层                     │
          │  主: Claude claude-sonnet-4-6  备: GPT-4o-mini │
          │  (带重试 + 指数退避 + 熔断)                │
          └──────────────────┬────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │            可观测性层                       │
          │  LangSmith(链路) + Prometheus(指标)        │
          │  Grafana(仪表盘) + AlertManager(告警)      │
          └───────────────────────────────────────────┘
          
          ┌──────────────────────────────────────────┐
          │          基础设施层                        │
          │  PostgreSQL(用户记忆) SQLite(会话状态)     │
          │  Docker Compose 容器编排                   │
          └──────────────────────────────────────────┘
```

---

## 二、LangGraph Agent 设计

### 2.1 状态定义

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
    order_context: Optional[dict]             # 当前讨论的订单信息
    
    # 执行控制
    needs_human_transfer: bool                # 是否需要转人工
    transfer_reason: str                      # 转人工原因
    
    # 监控
    token_used: int                           # 本次调用 token 数
    response_time_ms: int                     # 响应时间
    quality_score: Optional[int]              # 自动评估分数
```

### 2.2 Graph 结构

```python
# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode

def build_customer_service_graph():
    workflow = StateGraph(CustomerServiceState)

    # 节点
    workflow.add_node("memory_loader",    memory_loader_node)   # 加载用户记忆
    workflow.add_node("context_trimmer",  context_trimmer_node) # Token 窗口管理
    workflow.add_node("agent",            agent_node)           # 主 LLM 推理
    workflow.add_node("tools",            ToolNode(cs_tools))   # 工具执行
    workflow.add_node("quality_checker",  quality_checker_node) # 质量评估
    workflow.add_node("memory_saver",     memory_saver_node)    # 保存长期记忆
    workflow.add_node("human_transfer",   human_transfer_node)  # 转人工

    # 入口
    workflow.set_entry_point("memory_loader")
    
    # 固定边
    workflow.add_edge("memory_loader",   "context_trimmer")
    workflow.add_edge("context_trimmer", "agent")
    workflow.add_edge("tools",           "agent")
    workflow.add_edge("quality_checker", "memory_saver")
    workflow.add_edge("memory_saver",    END)
    workflow.add_edge("human_transfer",  END)

    # 条件路由
    workflow.add_conditional_edges(
        "agent",
        route_agent_output,
        {
            "tool_call":       "tools",
            "human_transfer":  "human_transfer",
            "respond":         "quality_checker",
        }
    )
    
    checkpointer = SqliteSaver.from_conn_string("sessions.db")
    return workflow.compile(checkpointer=checkpointer)

def route_agent_output(state: CustomerServiceState) -> str:
    last_msg = state["messages"][-1]
    
    if state["needs_human_transfer"]:
        return "human_transfer"
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_call"
    return "respond"
```

---

## 三、记忆系统设计

### 3.1 短期记忆：Token 窗口管理

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

### 3.2 长期记忆：跨会话用户记忆

```python
# memory/long_term.py
from mem0 import Memory  # 使用 Mem0 框架（可选：自研）
import asyncpg

class UserMemoryManager:
    """
    跨会话用户记忆管理
    - 存储：PostgreSQL + pgvector（语义检索）
    - 读取：对话开始时语义召回
    - 写入：对话结束后自动提炼
    """
    
    def __init__(self):
        self.mem0 = Memory.from_config({
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "host": os.getenv("PG_HOST"),
                    "dbname": "customer_memories"
                }
            },
            "llm": {
                "provider": "anthropic",
                "config": {"model": "claude-haiku-4-5-20251001"}  # 用小模型做记忆操作
            }
        })
    
    async def load_memories(self, user_id: str, current_query: str) -> List[str]:
        """对话开始时，召回与当前问题相关的用户记忆"""
        results = self.mem0.search(
            query=current_query,
            user_id=user_id,
            limit=5
        )
        return [r["memory"] for r in results["results"]]
    
    async def save_memories(self, user_id: str, conversation: list):
        """
        对话结束后，从对话中提炼需要长期记忆的信息
        Mem0 会自动判断哪些内容值得记忆，避免垃圾信息
        """
        messages_for_mem0 = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant",
             "content": m.content}
            for m in conversation if hasattr(m, "content")
        ]
        
        self.mem0.add(
            messages_for_mem0,
            user_id=user_id,
            metadata={"source": "customer_service", "timestamp": datetime.now().isoformat()}
        )
    
    async def delete_user_memories(self, user_id: str):
        """GDPR 合规：用户要求删除记忆"""
        self.mem0.delete_all(user_id=user_id)


# 节点实现
async def memory_loader_node(state: CustomerServiceState) -> dict:
    manager = UserMemoryManager()
    
    # 获取最新用户消息
    last_human_msg = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    
    memories = await manager.load_memories(state["user_id"], last_human_msg)
    
    return {"user_memories": memories}

async def memory_saver_node(state: CustomerServiceState) -> dict:
    """对话结束后保存记忆（异步，不影响响应速度）"""
    manager = UserMemoryManager()
    await manager.save_memories(state["user_id"], state["messages"])
    return {}
```

---

## 四、限流设计

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

## 五、LLM 降级与重试

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

## 六、监控与可观测性

### 6.1 LangSmith 链路追踪

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

### 6.2 Prometheus 指标

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

### 6.3 自动质量评估

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

## 七、MCP 集成设计

### 7.1 业务工具 MCP Server

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

### 7.2 MCP 配置

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

## 八、System Prompt 工程设计

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

## 九、Docker Compose 部署

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

## 十、目录结构

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
│   ├── tools.py                 # 工具定义
│   └── prompts.py               # Prompt 模板
├── memory/
│   ├── short_term.py            # Token 窗口管理
│   └── long_term.py             # Mem0 长期记忆
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

## 十一、性能基准测试

```python
# tests/load_test.py - 使用 locust
from locust import HttpUser, task, between

class CustomerServiceUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def ask_order_status(self):
        self.client.post("/chat", json={
            "user_id": "test_user_001",
            "message": "我的订单ORD123456到哪了？"
        })
    
    @task(2)
    def ask_product_info(self):
        self.client.post("/chat", json={
            "user_id": "test_user_002",
            "message": "这个商品支持7天无理由退货吗？"
        })
    
    @task(1)
    def request_refund(self):
        self.client.post("/chat", json={
            "user_id": "test_user_003",
            "message": "我要申请退款，商品有质量问题"
        })

# 运行命令: locust -f tests/load_test.py --host=http://localhost:8000
# 目标: 50并发用户，平均响应时间 < 5s，错误率 < 1%
```
