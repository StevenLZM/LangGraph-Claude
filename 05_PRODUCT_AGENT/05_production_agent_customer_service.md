# PRD：生产级 AI Agent 平台 — 智能客服系统

> 项目编号：05 | 难度：⭐⭐⭐⭐⭐ | 预计周期：4 周

---

## 一、项目背景与目标

### 背景

前四个项目解决了"能跑起来"的问题，但生产环境中 AI Agent 面临更严峻的挑战：高并发下如何保证稳定性？如何防止 Token 超支？如何追踪每次对话的质量？如何在系统崩溃后快速恢复？本项目将一个智能客服系统做到"生产级"标准。

### 目标

构建一个可承接真实流量的智能客服 Agent，具备完整的监控、限流、记忆管理、错误恢复和可观测性能力，达到上线运营标准。

### 学习目标

| 技能点 | 掌握内容 |
|--------|----------|
| 长期记忆 | 跨会话用户记忆存储与召回 |
| 短期记忆 | 对话窗口管理，防止 Context 溢出 |
| 限流 | Token 用量限制、QPS 控制 |
| 监控 | LangSmith Tracing + 自定义指标 |
| 错误恢复 | 重试策略、降级方案、熔断机制 |
| 并发 | 异步处理多用户并发请求 |
| 评估 | 自动化评估 Agent 回答质量 |
| 部署 | Docker 容器化 + 环境变量管理 |

---

## 二、用户故事

```
作为一名电商平台运营负责人
我想部署一个 AI 客服系统
能记住每位用户的历史订单和偏好
并发处理数百个对话不崩溃
当 AI 无法处理时自动转人工
所有对话都有质量监控和成本记录
以便我能衡量 ROI、及时发现问题
```

---

## 三、功能需求

### 3.1 核心对话能力

- **F01** 支持多轮对话，理解上下文（如"我的订单"中的"我的"）
- **F02** 可查询订单状态、物流信息、商品详情（接入 Mock API）
- **F03** 处理常见问题：退换货、投诉建议、账户问题
- **F04** 无法处理时，无缝转接人工客服（标记转接原因）
- **F05** 支持中英文混合输入

### 3.2 记忆管理（核心）

#### 短期记忆（对话级）

- **F06** 维护当前对话的消息窗口（最近 N 条）
- **F07** 当对话超过窗口时，自动压缩早期内容（摘要化）
- **F08** 确保 Prompt 总 Token 数不超过模型限制的 80%

#### 长期记忆（用户级）

- **F09** 跨会话记住用户信息：姓名、偏好、历史投诉记录
- **F10** 关键事件自动提炼存储：如"用户对物流极度不满"
- **F11** 对话开始时自动召回相关用户记忆（语义检索）
- **F12** 用户可要求删除其记忆数据（隐私合规）

### 3.3 限流与配额

- **F13** 单用户每分钟最多 10 次请求（防刷）
- **F14** 单次对话最大 Token 消耗：4000 tokens
- **F15** 全局每小时 Token 预算，超出后降级为简化回答
- **F16** 限流时返回友好提示而非报错

### 3.4 监控与可观测性

- **F17** 每次对话全链路追踪（LangSmith Tracing）
- **F18** 实时仪表盘显示：QPS、平均响应时间、Token 用量、错误率
- **F19** 自动评估每次回答质量（准确性、礼貌性、完整性）
- **F20** 低质量回答（评分 < 70）触发告警
- **F21** 成本统计：每次对话的 Token 费用

### 3.5 错误恢复

- **F22** API 调用失败自动重试（指数退避，最多 3 次）
- **F23** 主 LLM 不可用时，降级到备用模型
- **F24** 所有错误记录到日志，包含完整上下文
- **F25** 服务重启后，进行中的对话状态自动恢复

---

## 四、技术架构

```
                        负载均衡
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         用户A对话      用户B对话      用户C对话
              │             │             │
              └─────────────┼─────────────┘
                            │
                     FastAPI 服务层
                    (异步处理并发请求)
                            │
                    ┌───────┴────────┐
                    │                │
              限流检查器         记忆管理器
              (Redis 计数)     (Mem0 / 自研)
                    │                │
                    └───────┬────────┘
                            │
                     LangGraph Agent
                    ┌───────┴────────┐
                    │                │
              短期记忆窗口      工具调用层
              (Token 管理)    (订单/物流API)
                    │                │
                    └───────┬────────┘
                            │
                        LLM 调用层
                    ┌───────┴────────┐
                    │                │
              主模型(Claude)    备用模型(GPT)
                            │
                    可观测性层
                    ┌───────┴────────┐
                    │                │
                LangSmith         Prometheus
                (链路追踪)         (指标采集)
                            │
                       Grafana 仪表盘
```

### 技术选型

| 层次 | 技术 | 说明 |
|------|------|------|
| API 框架 | FastAPI + uvicorn | 异步支持高并发 |
| Agent 框架 | LangGraph | 状态管理与工作流 |
| 短期记忆 | LangGraph MemorySaver | 对话窗口管理 |
| 长期记忆 | Mem0 / Chroma + 自定义 | 跨会话用户记忆 |
| 限流 | Redis + 令牌桶算法 | QPS 和 Token 配额 |
| 监控 | LangSmith + Prometheus | 链路 + 指标 |
| 可视化 | Grafana | 运营仪表盘 |
| 主 LLM | claude-sonnet​​​-4-6 | 生产主力 |
| 备用 LLM | GPT-4o-mini | 降级备用 |
| 数据库 | PostgreSQL | 用户记忆持久化 |
| 缓存 | Redis | 限流 + 热点缓存 |
| 容器化 | Docker + docker-compose | 一键部署 |

---

## 五、核心实现要点

### 5.1 Token 管理（防止 Context 溢出）

```python
from langchain_core.messages import SystemMessage, trim_messages

def manage_context_window(messages: list, max_tokens: int = 3200):
    """保留最近的消息，超出时压缩早期内容"""
    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",          # 保留最新消息
        token_counter=llm,        # 使用 LLM 计算 token
        include_system=True,      # 保留 system prompt
        allow_partial=False
    )
    return trimmed
```

### 5.2 长期记忆召回

```python
async def load_user_memory(user_id: str, current_query: str) -> str:
    """对话开始时，召回与当前话题相关的用户记忆"""
    
    # 语义检索相关记忆
    relevant_memories = await memory_store.search(
        namespace=f"user:{user_id}",
        query=current_query,
        top_k=5
    )
    
    if not relevant_memories:
        return ""
    
    memory_text = "\n".join([m.content for m in relevant_memories])
    return f"关于该用户的背景信息：\n{memory_text}"

async def save_conversation_memory(user_id: str, conversation: list):
    """对话结束时，提炼关键信息存入长期记忆"""
    summary = await llm.ainvoke(
        f"提炼以下对话中需要长期记住的用户信息（投诉、偏好、重要事件）：\n{conversation}"
    )
    await memory_store.put(
        namespace=f"user:{user_id}",
        content=summary.content,
        metadata={"timestamp": datetime.now().isoformat()}
    )
```

### 5.3 限流实现

```python
import redis.asyncio as redis
from fastapi import HTTPException

async def check_rate_limit(user_id: str, redis_client):
    key = f"ratelimit:{user_id}:{int(time.time() // 60)}"
    
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    results = await pipe.execute()
    
    request_count = results[0]
    if request_count > 10:  # 每分钟 10 次
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试"
        )
```

### 5.4 LLM 降级策略

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def call_llm_with_fallback(messages: list) -> str:
    try:
        # 优先使用主模型
        response = await primary_llm.ainvoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"主模型调用失败: {e}, 切换备用模型")
        # 降级到备用模型
        response = await fallback_llm.ainvoke(messages)
        return response.content
```

### 5.5 自动质量评估

```python
async def evaluate_response(question: str, answer: str) -> dict:
    """使用 LLM 自动评估回答质量"""
    eval_prompt = f"""
    评估以下客服回答的质量（0-100分）：
    
    用户问题：{question}
    客服回答：{answer}
    
    评估维度：
    1. 准确性（是否正确回答了问题）
    2. 完整性（是否全面覆盖了问题）
    3. 礼貌性（语气是否专业友好）
    
    输出JSON格式：{{"score": 85, "issues": ["..."], "passed": true}}
    """
    result = await evaluator_llm.ainvoke(eval_prompt)
    return json.loads(result.content)
```

---

## 六、监控仪表盘指标

```
┌─────────────────────────────────────────────────────┐
│  智能客服运营仪表盘                    实时更新        │
├──────────────┬──────────────┬───────────────────────┤
│  当前 QPS    │  平均响应时间  │   今日 Token 费用      │
│    23.5      │    2.3s      │      $12.45           │
├──────────────┼──────────────┼───────────────────────┤
│  对话成功率   │  转人工率     │   平均满意度评分        │
│   97.2%      │    8.3%      │      82.1 / 100       │
├──────────────┴──────────────┴───────────────────────┤
│  最近 1 小时错误分布                                  │
│  ▇▇▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁        │
└─────────────────────────────────────────────────────┘
```

---

## 七、评估标准

**功能验证**
- [ ] 100 轮对话后记忆正常，无 Context 溢出崩溃
- [ ] 限流机制生效：第 11 次请求返回 429
- [ ] 主模型断开后自动切换备用模型
- [ ] LangSmith 中可查看每次对话完整链路

**性能验证**
- [ ] 并发 50 用户，平均响应时间 ≤ 5 秒
- [ ] 连续运行 24 小时无崩溃

**质量验证**
- [ ] 100 个测试问题，自动评估平均分 ≥ 80
- [ ] 低质量回答告警成功触发

---

## 八、项目交付物

```
project/
├── api/
│   ├── main.py          # FastAPI 入口
│   ├── routers/         # 路由处理
│   └── middleware/      # 限流、认证中间件
├── agent/
│   ├── graph.py         # LangGraph 工作流
│   ├── nodes.py         # 各节点实现
│   └── tools.py         # 工具定义
├── memory/
│   ├── short_term.py    # 对话窗口管理
│   └── long_term.py     # 跨会话记忆
├── monitoring/
│   ├── tracing.py       # LangSmith 配置
│   ├── metrics.py       # Prometheus 指标
│   └── evaluator.py     # 质量自动评估
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── grafana/         # Grafana 仪表盘配置
├── tests/
│   ├── load_test.py     # 并发压力测试
│   └── eval_dataset.json # 评估数据集
└── README.md
```

---

## 九、扩展方向

- **多租户支持**：不同客户隔离数据和配额
- **A/B 测试**：不同 Prompt 版本效果对比
- **主动学习**：从人工修正案例中优化 Prompt
- **语音接入**：支持电话客服（TTS/STT 集成）
- **情感检测**：识别愤怒用户，优先转人工处理
