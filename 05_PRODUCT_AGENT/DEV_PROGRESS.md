# 开发进度日志（DEV_PROGRESS）

> 本文档是 05_PRODUCT_AGENT 的工程进度入口。后续开发都在 `main` 分支进行，并以本文档记录迭代目标、验收状态、关键决策和未竟事项。
>
> 最后更新：2026-05-05
> 当前阶段：**M2 待启动**（M1 客服对话 MVP 与 UI 已完成）

---

## 一、项目速查

| 维度 | 内容 |
|---|---|
| 项目 | 05_PRODUCT_AGENT |
| 名称 | 生产级 AI Agent 平台 —— 智能客服系统 |
| 定位 | 面向真实流量的生产级客服 Agent，重点验证并发、记忆、限流、成本、监控、降级、评估和部署能力 |
| PRD | `05_production_agent_customer_service.md` |
| 工程设计 | `05_production_engineering.md` |
| 当前代码状态 | 已完成 M1：FastAPI `/chat`、内置客服 UI、Mock 订单/物流/商品/退款工具、规则型客服 Agent、pytest 测试 |
| 开发分支 | `main` |
| API 默认端口 | `8000` |
| 主要技术栈 | FastAPI、LangGraph、Redis、PostgreSQL/pgvector、Mem0 或 Chroma、LangSmith、Prometheus、Grafana、Docker Compose |

05 的目标不是再证明 Agent 能“跑起来”，而是把 Agent 放到真实客服系统里，具备上线运营所需的稳定性、成本控制、质量追踪和恢复能力。

---

## 二、迭代路线

### M0 项目骨架（已完成）

**目标**
- 建立可运行的工程基础，使后续每个能力都能通过测试和接口逐步接入。

**主要交付**
- 目录结构：`api/`、`agent/`、`memory/`、`llm/`、`monitoring/`、`mcp_servers/`、`infra/`、`tests/`
- FastAPI 入口：`GET /health`
- 基础配置：`.env.example`、`requirements.txt`
- 测试框架：pytest 能运行最小测试集
- LangGraph 空图或最小图：可构建、可调用、可返回固定响应

**验收标准**
- `pytest tests -q` 通过
- `uvicorn api.main:app --host 0.0.0.0 --port 8000` 可启动
- `GET /health` 返回健康状态

### M1 客服对话 MVP（已完成）

**目标**
- 跑通单用户智能客服闭环：用户提问、Agent 判断、必要时调用业务工具、返回客服回复或转人工标记。

**主要交付**
- `POST /chat`
- `CustomerServiceState` 基础字段：`session_id`、`user_id`、`messages`、`order_context`、`needs_human_transfer`、`transfer_reason`
- 客服 System Prompt：身份、能力边界、退款确认、转人工规则
- Mock 业务工具：订单查询、物流查询、商品查询、退款申请
- LangGraph 主流程：`agent -> tools -> agent -> respond/human_transfer`
- FastAPI 内置 UI：`GET /` 客服工作台，支持对话、快捷问题、订单上下文和转人工状态展示

**验收标准**
- 能回答订单状态、物流状态、商品咨询
- 退款申请必须在用户明确同意后才执行
- 用户要求人工、投诉纠纷、法律问题等场景能标记转人工
- 中英文混合输入不导致接口失败

### M2 记忆系统（待启动）

**目标**
- 做出 05 项目的核心差异：同时具备短期上下文管理和跨会话用户长期记忆。

**主要交付**
- 短期记忆：Token 窗口裁剪、早期对话摘要、保留最近 8 轮
- 会话状态：SQLite checkpointer 或等价持久化机制
- 长期记忆：PostgreSQL/pgvector + Mem0，或第一版使用 Chroma 但保持接口稳定
- 记忆接口：加载用户记忆、保存关键事件、删除用户记忆
- `DELETE /users/{user_id}/memories`

**验收标准**
- 100 轮对话后不发生 Context 溢出
- 新会话能召回用户偏好、投诉记录等历史信息
- 用户要求删除记忆后，后续对话不能再召回被删除内容
- 服务重启后，会话状态可恢复或可查询

### M3 限流与弹性（待启动）

**目标**
- 控制成本和失败影响，让系统在高频请求、Token 超支、LLM 异常时仍能给出可控响应。

**主要交付**
- Redis 用户级限流：单用户每分钟 10 次请求
- 全局 QPS 控制：默认每秒 100 QPS
- Token 预算：单次对话 4000 tokens，全局每小时 500000 tokens
- 预算超限降级：简化回答或切换便宜模型
- LLM 弹性层：指数退避重试、主备模型切换、熔断器

**验收标准**
- 同一用户第 11 次/分钟请求返回 429，并包含友好提示
- 主模型失败后自动切换备用模型
- 连续失败达到阈值后熔断器开启，短期内优先走备用模型
- Token 预算超限不导致服务崩溃

### M4 可观测性与质量评估（待启动）

**目标**
- 让系统可运营：每次对话可追踪，关键指标可监控，低质量回答可发现。

**主要交付**
- LangSmith tracing：按 `session_id`、`user_id` 打 tag 和 metadata
- Prometheus 指标：请求数、响应时间、Token 消耗、活跃会话数、质量评分
- `GET /metrics`
- 自动质量评估：准确性、礼貌性、完整性，加权总分
- 低质量告警：评分低于 70 记录告警事件

**验收标准**
- LangSmith 中可查看每次对话完整链路
- Prometheus 可抓取指标
- Grafana 看板展示 QPS、平均响应时间、Token 用量、错误率、转人工率、质量评分
- 低质量回答能触发告警记录

### M5 部署与压测（待启动）

**目标**
- 达到本地一键部署和性能演示标准。

**主要交付**
- Dockerfile
- Docker Compose：`api`、`redis`、`postgres/pgvector`、`prometheus`、`grafana`
- Locust 压测脚本
- README：安装、配置、启动、压测、监控访问方式

**验收标准**
- `docker compose up --build` 一键启动
- API、Redis、Postgres、Prometheus、Grafana 均正常运行
- 50 并发用户平均响应时间不超过 5 秒
- 压测错误率低于 1%
- 连续运行 24 小时无崩溃

### M6 收尾强化（待启动）

**目标**
- 补齐作品集和真实项目质感，形成可演示、可评估、可接续优化的生产级 Agent 项目。

**主要交付**
- 管理接口：会话列表、用户记忆查看、转人工统计
- 评估数据集：100 个客服问题
- 自动评测脚本和报告
- FAQ/RAG 工具接入：可复用 01_RAG 的知识库能力
- 故障排查文档和运行手册

**验收标准**
- 100 个客服问题自动评估平均分不低于 80
- 管理侧能查看会话、质量分、Token 成本和转人工原因
- README 能支持新开发者独立启动和演示项目

---

## 三、关键架构决策

### 1. 先客服闭环，再生产能力

M1 先完成客服业务闭环，避免一开始陷入监控、部署、限流等基础设施细节。后续 M2-M5 逐层加入生产能力，每个迭代都必须有独立可验收结果。

### 2. 记忆系统分短期和长期

短期记忆服务于当前对话，解决 Context 窗口问题；长期记忆服务于用户画像和跨会话体验，解决用户偏好、投诉记录、重要事件的保存与召回。两者不能混在一个消息列表里处理。

### 3. 成本控制是核心功能，不是运维附属

05 明确要求 Token 配额、QPS 控制、降级回答和成本统计。因此限流、预算和模型选择必须进入主流程，而不是仅作为外部网关能力。

### 4. LLM 失败是常态路径

主模型超时、限流、不可用都必须被视为正常生产风险。M3 需要把重试、备用模型、熔断器做成统一的 `ResilientLLM` 层，Agent 节点不直接调用裸模型。

### 5. 转人工是客服安全边界

转人工不是 04 项目的审批式 HITL，但它是客服系统的业务安全边界。用户明确要求人工、投诉纠纷、法律问题、连续无法解决、情绪激动时，Agent 必须停止硬答并标记原因。

### 6. 观测指标要面向运营

05 的监控不只看服务是否活着，还要看 QPS、延迟、Token 成本、错误率、转人工率、质量评分。最终指标应能回答：系统是否稳定、是否省钱、是否真的解决用户问题。

---

## 四、计划中的代码组织

```
05_PRODUCT_AGENT/
├── 05_production_agent_customer_service.md
├── 05_production_engineering.md
├── DEV_PROGRESS.md
├── README.md
├── requirements.txt
├── .env.example
│
├── api/
│   ├── main.py                  # FastAPI 入口 + health + chat + UI
│   ├── schemas.py               # API 请求/响应模型
│   ├── ui.py                    # 内置客服工作台 HTML
│   ├── routers/
│   │   └── admin.py             # 管理接口（后续）
│   └── middleware/
│       ├── rate_limiter.py      # 用户限流、QPS、Token 预算
│       └── auth.py              # 预留认证
│
├── agent/
│   ├── graph.py                 # LangGraph 工作流
│   ├── state.py                 # CustomerServiceState
│   ├── nodes.py                 # memory/context/agent/quality/handoff 节点
│   ├── service.py               # 规则型客服决策
│   ├── intent.py                # 意图与订单号识别
│   ├── tools.py                 # Mock 业务工具定义
│   └── prompts.py               # 客服 System Prompt
│
├── memory/
│   ├── short_term.py            # Token 窗口裁剪与摘要
│   └── long_term.py             # 用户长期记忆
│
├── llm/
│   └── resilient_llm.py         # 主备模型、重试、熔断
│
├── mcp_servers/
│   ├── order_server.py          # 订单/物流 MCP Server
│   ├── kb_server.py             # FAQ/RAG MCP Server
│   └── crm_server.py            # CRM MCP Server
│
├── monitoring/
│   ├── tracing.py               # LangSmith 配置
│   ├── metrics.py               # Prometheus 指标
│   └── evaluator.py             # 自动质量评估
│
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   └── grafana/
│
└── tests/
    ├── test_chat_flow.py
    ├── test_tools.py
    ├── test_ui.py
    ├── test_rate_limiter.py
    ├── test_memory.py
    ├── test_llm_fallback.py
    ├── test_quality_evaluator.py
    └── load_test.py
```

---

## 五、接口基线

### `POST /chat`

输入：
```json
{
  "user_id": "user_001",
  "session_id": "session_001",
  "message": "我的订单 ORD123456 到哪了？"
}
```

输出：
```json
{
  "session_id": "session_001",
  "user_id": "user_001",
  "answer": "订单 ORD123456 的物流由顺丰承运，运单号 SF100200300CN。最新状态：快件已到达配送站，正在安排派送，当前位置：上海浦东配送站。",
  "needs_human_transfer": false,
  "transfer_reason": "",
  "order_context": {
    "order_id": "ORD123456",
    "status": "配送中",
    "product": "AirBuds Pro 2"
  },
  "token_used": 9,
  "response_time_ms": 0,
  "quality_score": 88
}
```

### `GET /`

返回内置客服工作台 UI，可直接在浏览器中试用订单、物流、商品、退款和转人工场景。

### `GET /sessions/{session_id}`

查询会话状态、最近消息窗口、是否转人工、当前 Token 使用情况。

### `DELETE /users/{user_id}/memories`

删除指定用户长期记忆，删除后新会话不得再召回旧记忆。

### `GET /health`

返回 API、Redis、数据库、LLM 配置的基础健康状态。

### `GET /metrics`

Prometheus 指标出口。

---

## 六、当前状态

- [x] 已完成 PRD：`05_production_agent_customer_service.md`
- [x] 已完成工程设计：`05_production_engineering.md`
- [x] 已明确后续开发都在 `main` 分支进行
- [x] 已建立本进度文档
- [x] M0 项目骨架
- [x] M1 客服对话 MVP
- [ ] M2 记忆系统
- [ ] M3 限流与弹性
- [ ] M4 可观测性与质量评估
- [ ] M5 部署与压测
- [ ] M6 收尾强化

---

## 七、迭代记录

### 2026-05-05：M0 项目骨架完成

**实际交付**
- 新增 `api/`：FastAPI 入口、`GET /health`、基础配置读取。
- 新增 `agent/`：`CustomerServiceState`、最小 LangGraph 工作流、离线固定客服回复节点。
- 新增 `memory/`、`llm/`、`monitoring/`、`mcp_servers/`、`infra/`：后续迭代占位模块。
- 新增 `.env.example`、`requirements.txt`、`README.md`。
- 新增 `tests/`：覆盖健康检查、配置默认值、图编译和离线调用。

**验收结果**
- `pytest 05_PRODUCT_AGENT/tests -q`：4 passed。
- `cd 05_PRODUCT_AGENT && pytest tests -q`：4 passed。
- `cd 05_PRODUCT_AGENT && uvicorn api.main:app --host 127.0.0.1 --port 8000`：可启动。
- `GET /health`：返回 `status=ok`、`graph_ready=true`、`llm=offline_stub`。

**遗留到 M1**
- `/chat` 尚未实现。
- 业务工具仍为空占位，订单、物流、商品、退款 Mock 数据将在 M1 接入。
- 当前图使用离线固定回复，不调用真实 LLM。

---

### 2026-05-05：M1 客服对话 MVP 与 UI 完成

**实际交付**
- 新增 `POST /chat`：输入 `user_id`、`session_id`、`message`，返回客服回答、转人工标记、订单上下文、Token 估算、质量分占位。
- 新增 `GET /`：FastAPI 内置客服工作台 UI，包含快捷问题、聊天区、运行状态、订单上下文展示。
- 新增 Mock 业务工具：`get_order`、`get_logistics`、`get_product`、`apply_refund`。
- 新增规则型客服决策：订单查询、物流查询、商品库存、退款二次确认、投诉/法律/人工诉求转人工、中英文混合订单查询。
- 新增 M1 测试：`test_chat_flow.py`、`test_tools.py`、`test_ui.py`。

**验收结果**
- `pytest 05_PRODUCT_AGENT/tests -q`：17 passed。
- `/chat` 覆盖订单、物流、商品、退款确认、转人工和中英文混合输入。
- `/` 返回客服工作台 HTML，可通过浏览器访问。

**遗留到 M2**
- 会话历史仍未持久化，M2 需要引入短期记忆窗口和会话状态恢复。
- 当前客服决策是离线规则实现，真实 LLM 与工具调用路由留到后续迭代接入。
- 质量分和 Token 用量仍是轻量占位，M4 需要接入正式评估和 Prometheus 指标。

---

## 八、后续开发约定

1. 每个迭代完成后，必须更新本文档的当前阶段、已完成项、关键决策和遗留问题。
2. 每个迭代必须保持一个可运行验收点，避免只堆代码不可演示。
3. 先稳定接口，再增强内部实现；`/chat`、`/health`、`/metrics` 应尽早固定。
4. 测试覆盖随风险增加而增加：M0/M1 重接口闭环，M2 重记忆一致性，M3 重异常路径，M4/M5 重观测和压测。
5. 不把真实密钥、用户隐私数据、生产订单数据写入仓库。
6. 与 01_RAG、03_MULTI_AGENT 的复用只通过清晰适配层完成，避免互相污染 import 路径和配置。

---

## 九、未竟事项

- 选择长期记忆第一版实现：Mem0 + pgvector，或 Chroma 自研适配。
- 明确主模型和备用模型在本地环境的实际可用配置。
- 确定 Mock 订单、物流、商品、退款数据格式。
- 确定质量评估数据集的题型分布。
- 确定 Grafana 看板是否直接使用 JSON provisioning。
