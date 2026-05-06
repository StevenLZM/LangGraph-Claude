# 05_PRODUCT_AGENT

生产级 AI Agent 平台的智能客服项目。当前已完成 M2 记忆系统：`/chat`、内置客服工作台、Mock 业务工具、规则型客服 Agent、短期记忆窗口、SQLite 会话状态和用户长期记忆。

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

## 测试

```bash
pytest tests -q
```

M2 使用离线规则型客服决策和本地 SQLite 记忆存储，不调用真实 LLM，也不依赖 Redis、PostgreSQL 或外部密钥。

## 教学文档

- `Codex_TEACHING_GUIDE.md`：已开发部分的关键代码、流程对照解释和面试重点。
