# 05_PRODUCT_AGENT

生产级 AI Agent 平台的智能客服项目。当前已完成 M5 部署与压测：`/chat`、内置客服工作台、Mock 工具、规则型客服 Agent、短期记忆窗口、SQLite 会话和用户长期记忆、限流与 Token 预算降级、LLM fallback 与熔断测试层、LangSmith trace metadata、Prometheus 兼容指标、Grafana 看板编排和 Locust 压测入口。

## 本地运行

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
  -d '{"user_id":"user_001","session_id":"session_001","message":"我的订单 ORD123456 到哪了？"}'
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

## M5 Docker Compose 部署

复制配置并按需调整：

```bash
cp .env.example .env
```

一键启动 API、Redis、Postgres/pgvector、Prometheus、Grafana：

```bash
docker compose up --build
```

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
- `llm/resilient_llm.py` 提供可注入的主备模型、指数退避重试和熔断器；M3 保持客服主路径为离线规则型实现。

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

M5 默认仍使用离线规则型客服决策、本地 SQLite 记忆存储和确定性质量评估器；Compose 会启动 Redis、Postgres/pgvector、Prometheus 和 Grafana，但真实 LLM 主路径和 pgvector 记忆迁移仍留给后续迭代。

## 教学文档

- `Codex_TEACHING_GUIDE.md`：已开发部分的关键代码、流程对照解释和面试重点。
