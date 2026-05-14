# 05_PRODUCT_AGENT

生产级 AI Agent 平台的智能客服项目。当前已完成 M6 收尾强化、存储后端升级和 RocketMQ 业务消息接入：`/chat`、内置客服工作台、Mock 工具、规则型客服 Agent、短期记忆窗口、SQLite/Postgres 会话和用户长期记忆、LangGraph Postgres/Redis checkpointer、限流与 Token 预算降级、DeepSeek 真实 LLM 主路径、LLM fallback 与熔断测试层、FAQ/RAG 适配、RocketMQ 跨项目业务消息、管理接口、100 题自动评测、LangSmith trace metadata、Prometheus 兼容指标、Grafana 看板编排和 Locust 压测入口。

## 本地运行

`/chat` 现在只走真实 LLM 路径。启动前先复制配置并填入 DeepSeek key：

```bash
cd 05_PRODUCT_AGENT
cp .env.example .env
# 编辑 .env，设置 DEEPSEEK_API_KEY=sk-...
```

```bash
cd 05_PRODUCT_AGENT
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

客服工作台：

```text
http://127.0.0.1:8000/
```

对话接口：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user_001","session_id":"session_001","request_id":"req_001","message":"我的订单 ORD123456 到哪了？"}'
```

客户端重试同一条消息时，请复用同一个 `request_id`。服务端会按 `(user_id, session_id, request_id)` 做幂等去重，完成后的重复请求会返回第一次结果，并将 `request_status` 标记为 `replayed`。

请求状态查询：

```bash
curl "http://127.0.0.1:8000/chat/requests/req_001?user_id=user_001&session_id=session_001"
```

查询会话：

```bash
curl http://127.0.0.1:8000/sessions/session_001
```

删除用户记忆：

```bash
curl -X DELETE http://127.0.0.1:8000/users/user_001/memories
```

Prometheus 指标：

```bash
curl http://127.0.0.1:8000/metrics
```

管理接口：

```bash
curl http://127.0.0.1:8000/admin/sessions
curl http://127.0.0.1:8000/admin/users/user_001/memories
curl http://127.0.0.1:8000/admin/stats/transfers
```

## M6 DeepSeek 与 FAQ/RAG

运行时默认使用 DeepSeek 的 OpenAI-compatible API；`offline_stub` 已禁用，真实 LLM 未配置或调用失败时 `/chat` 会返回 `503 llm_unavailable`，不会静默回落到规则答案：

```bash
LLM_MODE=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MAX_MODEL=deepseek-v4-pro
DEEPSEEK_LIGHT_MODEL=deepseek-v4-flash
```

FAQ/RAG 问题会通过 `rag/faq_tool.py` 尝试复用兄弟项目 `01_RAG` 的 hybrid retriever。若 01 索引或依赖不可用，接口不会崩溃，会返回明确的知识库未命中/不可用结果。

自动评测：

```bash
python evals/run.py --dataset evals/dataset.jsonl
```

结果写入 `evals/results/<run_id>/results.jsonl` 和 `REPORT.md`。M6 数据集共 100 题，覆盖订单、物流、商品、退款、转人工、记忆、FAQ/RAG 和降级兜底场景。

## M5 Docker Compose 部署

复制配置并按需调整：

```bash
cp .env.example .env
```

一键启动 API、Redis、Postgres/pgvector、RocketMQ、Prometheus、Grafana：

```bash
docker compose up --build
```

Compose 默认会覆盖本地 `.env.example` 的轻量存储设置，使用 `STORAGE_BACKEND=postgres`、`CHECKPOINTER_BACKEND=postgres` 和 `ROCKETMQ_ENABLED=true`。Postgres 同时承载业务会话/用户记忆表和 LangGraph checkpoint 表；Redis 用于限流计数，也可通过 `CHECKPOINTER_BACKEND=redis` 作为可选 checkpoint 后端；RocketMQ 用于跨项目业务消息。

如果要在 LangSmith 网站看到当前项目，先在 `.env` 中配置：

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=production-agent-customer-service
```

启动后至少发送一次 `/chat` 请求，LangSmith 才会创建/显示对应项目 run。`/health` 和 `/metrics` 不会产生 LangGraph trace。

服务入口：

- API：`http://127.0.0.1:8000`
- 客服工作台：`http://127.0.0.1:8000/`
- Prometheus：`http://127.0.0.1:9090`
- Grafana：`http://127.0.0.1:3000`，默认账号密码为 `.env` 中的 `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`

停止并保留数据卷：

```bash
docker compose down
```

清理容器和数据卷：

```bash
docker compose down -v
```

## M3 限流与弹性

默认不需要 Redis：`REDIS_URL` 为空时使用进程内存限流，适合本地演示和 pytest。配置 `REDIS_URL=redis://localhost:6379` 后会使用 Redis 计数。

默认策略：

- 单用户每分钟最多 `10` 次 `/chat` 请求，第 11 次返回 `429` 和友好提示。
- 全局每秒最多 `100` 次 `/chat` 请求，超限返回 `503`。
- 单次对话预算 `4000` tokens，全局每小时预算 `500000` tokens。
- Token 预算超限不崩溃，`/chat` 返回简化回复，并设置 `degraded=true` 和 `degrade_reason`。
- `llm/resilient_llm.py` 提供可注入的主备模型、指数退避重试和熔断器；当前 `/chat` 正常客服回答必须通过真实 LLM。

## 存储后端

本地默认 `STORAGE_BACKEND=sqlite`、`CHECKPOINTER_BACKEND=none`，会话和用户记忆写入 `MEMORY_DB`。Docker Compose 默认使用 Postgres：

- `STORAGE_BACKEND=postgres`：`SessionStore` 和 `UserMemoryManager` 写入 `DATABASE_URL`。
- `CHECKPOINTER_BACKEND=postgres`：LangGraph 原生 checkpoint 写入 `DATABASE_URL`，首次启动会执行 `setup()`。
- `CHECKPOINTER_BACKEND=redis`：可选使用 Redis checkpoint，适合短期可恢复状态；长期业务记忆仍建议使用 Postgres。
- `CHECKPOINTER_URL`：可覆盖 checkpoint 连接串；为空时 Postgres checkpoint 复用 `DATABASE_URL`，Redis checkpoint 复用 `REDIS_URL`。
- `CHECKPOINTER_SETUP=false`：适合生产环境已由迁移任务预建表/索引时关闭自动 `setup()`。

本地 SQLite 模式适合 pytest 和离线演示；Compose Postgres 模式更接近多实例部署。当前长期记忆仍是关键词召回，不是 pgvector 语义检索；后续如接 Mem0/pgvector，应复用现有 `UserMemoryManager` 接口，避免改动 `/chat` 和管理接口契约。

## RocketMQ 业务消息

本地默认 `ROCKETMQ_ENABLED=false`，不会要求启动 RocketMQ；Docker Compose 默认启用 RocketMQ nameserver/broker，展示生产级业务消息能力。`/chat` 在完成用户可见回复后发布两类消息：

- `ChatCompleted`：普通消息，topic `agent-customer-service-normal-v1`，用于跨项目订阅客服对话完成事件。
- `PostprocessRequested`：FIFO 顺序消息，topic `agent-customer-service-fifo-v1`，按 `session_id` 设置 message group，供质量评估、长期记忆提炼、运营统计等后处理消费者使用。
- `HumanTransferReminderRequested`：延迟消息，topic `agent-customer-service-delay-v1`，转人工场景下用于超时提醒或回访任务。

消息采用统一 envelope：`event_id`、`event_type`、`event_version`、`producer`、`occurred_at`、`trace_id`、`aggregate_id` 和 `payload`。生产端写入 `MESSAGE_OUTBOX_DB` 后再发送 RocketMQ，发送失败不会阻塞 `/chat`，管理接口可查看 outbox：

```bash
curl http://127.0.0.1:8000/admin/messages/outbox
```

后续接入其他项目时，建议按 RocketMQ 特性拆 topic：普通消息用于跨项目事件，FIFO 用于会话内顺序后处理，Delay 用于回访/超时提醒，Transaction 用于演示“本地业务写库 + 消息发送”的最终一致性。

## M4 可观测性与质量评估

`/chat` 现在会在 API 边界记录运营指标和质量评估结果，保持本地离线可测，不依赖外部 Prometheus Server、Grafana 或真实 LangSmith 服务。

- `GET /metrics` 暴露 Prometheus text exposition 格式指标：请求数、响应时间、Token 消耗、活跃会话数、错误数、转人工数和质量分。
- LangGraph 调用会注入 trace tags 与 metadata：`session_id`、`user_id`、`environment`、`version`。
- `AutoQualityEvaluator` 按准确性 40%、礼貌性 30%、完整性 30% 计算总分。
- 质量分低于 `QUALITY_ALERT_THRESHOLD`（默认 `70`）时记录结构化 warning 事件，并在会话 metadata 中标记 `quality_alert=true`。
- `LANGCHAIN_TRACING_V2=true` 且配置 `LANGCHAIN_API_KEY` 后，会写入 LangChain/LangSmith 相关环境变量；本地默认关闭。

## M5 压测

启动 Locust Web UI：

```bash
docker compose --profile loadtest up locust
```

打开 `http://127.0.0.1:8089`，Host 使用 `http://api:8000`。脚本覆盖订单、物流、商品、退款、转人工、健康检查和指标抓取场景。

无界面快速压测示例：

```bash
docker compose --profile loadtest run --rm locust \
  -f /mnt/locust/locustfile.py \
  --host http://api:8000 \
  --headless -u 50 -r 5 -t 2m
```

## 测试

```bash
pytest tests -q
```

M6 的业务 guardrail 仍由规则层负责，包括退款二次确认、转人工优先级和工具上下文构造；用户可见客服回答必须由真实 LLM 基于规则/工具结果生成。pytest 通过注入 fake LLM 保持离线稳定，不再依赖 `offline_stub` 作为运行模式。Compose 会启动 Redis、Postgres/pgvector、RocketMQ、Prometheus 和 Grafana；Postgres 业务存储、LangGraph checkpoint 和 RocketMQ outbox 已接入，pgvector/Mem0 语义记忆仍留给后续优化。

## 教学文档

- `Codex_TEACHING_GUIDE.md`：已开发部分的关键代码、流程对照解释和面试重点。
