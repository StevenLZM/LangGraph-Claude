# 05_PRODUCT_AGENT

生产级 AI Agent 平台的智能客服项目。当前已完成 M3 限流与弹性：`/chat`、内置客服工作台、Mock 工具、规则型客服 Agent、短期记忆窗口、SQLite 会话和用户长期记忆、用户级限流、全局 QPS 控制、Token 预算降级、LLM fallback 与熔断测试层。

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

## M3 限流与弹性

默认不需要 Redis：`REDIS_URL` 为空时使用进程内存限流，适合本地演示和 pytest。配置 `REDIS_URL=redis://localhost:6379` 后会使用 Redis 计数。

默认策略：

- 单用户每分钟最多 `10` 次 `/chat` 请求，第 11 次返回 `429` 和友好提示。
- 全局每秒最多 `100` 次 `/chat` 请求，超限返回 `503`。
- 单次对话预算 `4000` tokens，全局每小时预算 `500000` tokens。
- Token 预算超限不崩溃，`/chat` 返回简化回复，并设置 `degraded=true` 和 `degrade_reason`。
- `llm/resilient_llm.py` 提供可注入的主备模型、指数退避重试和熔断器；M3 保持客服主路径为离线规则型实现。

## 测试

```bash
pytest tests -q
```

M3 使用离线规则型客服决策和本地 SQLite 记忆存储，不调用真实 LLM，也不强制依赖 Redis、PostgreSQL 或外部密钥。

## 教学文档

- `Codex_TEACHING_GUIDE.md`：已开发部分的关键代码、流程对照解释和面试重点。
