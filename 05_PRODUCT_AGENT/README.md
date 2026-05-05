# 05_PRODUCT_AGENT

生产级 AI Agent 平台的智能客服项目。当前已完成 M1 客服对话 MVP：`/chat`、内置客服工作台、Mock 业务工具和规则型客服 Agent。

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

## 测试

```bash
pytest tests -q
```

M1 使用离线规则型客服决策，不调用真实 LLM，也不依赖 Redis、PostgreSQL 或外部密钥。

## 教学文档

- `Codex_TEACHING_GUIDE.md`：已开发部分的关键代码、流程对照解释和面试重点。
