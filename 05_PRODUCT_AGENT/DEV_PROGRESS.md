# 开发进度日志（DEV_PROGRESS）

> 本文档是 05_PRODUCT_AGENT 的工程进度入口。后续开发都在 `main` 分支进行，并以本文档记录迭代目标、验收状态、关键决策和未竟事项。
>
> 最后更新：2026-05-08
> 当前阶段：**M6 已完成**（DeepSeek 主路径、FAQ/RAG、管理接口与自动评测已接入）

---

## 一、项目速查

| 维度 | 内容 |
|---|---|
| 项目 | 05_PRODUCT_AGENT |
| 名称 | 生产级 AI Agent 平台 —— 智能客服系统 |
| 定位 | 面向真实流量的生产级客服 Agent，重点验证并发、记忆、限流、成本、监控、降级、评估和部署能力 |
| PRD | `05_production_agent_customer_service.md` |
| 工程设计 | `05_production_engineering.md` |
| 当前代码状态 | 已完成 M6：FastAPI `/chat`、内置客服 UI、Mock 工具、规则型客服 Agent、短期记忆窗口、SQLite 会话状态、用户长期记忆、Hybrid 限流、Token 预算降级、DeepSeek 真实 LLM 主路径、LLM fallback/熔断测试层、FAQ/RAG 适配、管理接口、100 题自动评测、Prometheus 兼容指标、LangSmith trace metadata、自动质量评估、Docker Compose、Grafana provisioning、Locust 压测入口、pytest 测试 |
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

### M2 记忆系统（已完成）

**目标**
- 做出 05 项目的核心差异：同时具备短期上下文管理和跨会话用户长期记忆。

**主要交付**
- 短期记忆：Token 窗口裁剪、早期对话摘要、保留最近 8 轮
- 会话状态：SQLite checkpointer 或等价持久化机制
- 长期记忆：第一版使用 SQLite 轻量实现，保持 `UserMemoryManager` 接口稳定，后续可替换为 Mem0 + pgvector
- 记忆接口：加载用户记忆、保存关键事件、删除用户记忆
- `GET /sessions/{session_id}`
- `DELETE /users/{user_id}/memories`

**验收标准**
- 100 轮对话后不发生 Context 溢出
- 新会话能召回用户偏好、投诉记录等历史信息
- 用户要求删除记忆后，后续对话不能再召回被删除内容
- 服务重启后，会话状态可恢复或可查询

### M3 限流与弹性（已完成）

**目标**
- 控制成本和失败影响，让系统在高频请求、Token 超支、LLM 异常时仍能给出可控响应。

**主要交付**
- Hybrid 用户级限流：有 `REDIS_URL` 时使用 Redis，未配置时使用内存；单用户每分钟 10 次请求
- 全局 QPS 控制：默认每秒 100 QPS
- Token 预算：单次对话 4000 tokens，全局每小时 500000 tokens
- 预算超限降级：返回简化回答，接口标记 `degraded` 和 `degrade_reason`
- LLM 弹性层：指数退避重试、主备模型切换、熔断器，可注入 fake client 测试

**验收标准**
- [x] 同一用户第 11 次/分钟请求返回 429，并包含友好提示
- [x] 主模型失败后自动切换备用模型
- [x] 连续失败达到阈值后熔断器开启，短期内优先走备用模型
- [x] Token 预算超限不导致服务崩溃

### M4 可观测性与质量评估（已完成）

**目标**
- 让系统可运营：每次对话可追踪，关键指标可监控，低质量回答可发现。

**主要交付**
- LangSmith tracing：按 `session_id`、`user_id` 打 tag 和 metadata
- Prometheus 指标：请求数、响应时间、Token 消耗、活跃会话数、质量评分
- `GET /metrics`
- 自动质量评估：准确性、礼貌性、完整性，加权总分
- 低质量告警：评分低于 70 记录告警事件

**验收标准**
- [x] LangSmith tracing 配置可携带 `session_id`、`user_id` metadata
- [x] Prometheus 可抓取 `GET /metrics` 指标
- [x] Grafana 所需的 QPS、平均响应时间、Token 用量、错误率、转人工率、质量评分指标已暴露
- [x] 低质量回答能触发告警记录

### M5 部署与压测（已完成）

**目标**
- 达到本地一键部署和性能演示标准。

**主要交付**
- Dockerfile
- Docker Compose：`api`、`redis`、`postgres/pgvector`、`prometheus`、`grafana`
- Locust 压测脚本
- README：安装、配置、启动、压测、监控访问方式

**验收标准**
- [x] `docker compose up --build` 一键启动配置已提供
- [x] API、Redis、Postgres/pgvector、Prometheus、Grafana 服务编排已提供
- [x] 50 并发用户 Locust 压测入口已提供
- [x] 压测错误率可由 Locust 报告观察
- [ ] 连续运行 24 小时待本机或 CI 环境执行长稳验证

### M6 收尾强化（已完成）

**目标**
- 补齐作品集和真实项目质感，形成可演示、可评估、可接续优化的生产级 Agent 项目。

**主要交付**
- 管理接口：会话列表、用户记忆查看、转人工统计
- 评估数据集：100 个客服问题
- 自动评测脚本和报告
- FAQ/RAG 工具接入：可复用 01_RAG 的知识库能力
- 故障排查文档和运行手册

**验收标准**
- [x] 100 个客服问题自动评估数据集和报告脚本已提供
- [x] 管理侧能查看会话、质量分、Token 成本和转人工原因
- [x] README 能支持新开发者独立启动、配置 DeepSeek、运行评测和演示项目

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
│   ├── session_store.py         # SQLite 会话状态
│   └── long_term.py             # SQLite 用户长期记忆
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
  "quality_score": 88,
  "user_memories": [],
  "memory_summary": "",
  "degraded": false,
  "degrade_reason": ""
}
```

预算超限时仍返回 `200`，但 `answer` 为简化回复，`degraded=true`，`degrade_reason` 为 `single_request_token_budget_exceeded` 或 `global_token_budget_exceeded`。

### `GET /`

返回内置客服工作台 UI，可直接在浏览器中试用订单、物流、商品、退款和转人工场景。

### `GET /sessions/{session_id}`

查询会话状态、最近消息窗口、是否转人工、当前 Token 使用情况和摘要。

### `DELETE /users/{user_id}/memories`

删除指定用户长期记忆，删除后新会话不得再召回旧记忆。

### `GET /health`

返回 API、Redis、数据库、LLM 和限流后端配置的基础健康状态。

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
- [x] M2 记忆系统
- [x] M3 限流与弹性
- [x] M4 可观测性与质量评估
- [x] M5 部署与压测
- [x] M6 收尾强化

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

### 2026-05-06：M2 记忆系统完成

**实际交付**
- 新增 `ContextWindowManager`：超过窗口时生成 `[早期对话摘要]`，保留最近 8 轮消息，并用轻量 token 估算控制上下文大小。
- 新增 `SessionStore`：使用 SQLite 持久化 `session_id`、`user_id`、消息窗口和会话元数据。
- 新增 `UserMemoryManager`：使用 SQLite 保存用户偏好、投诉记录和用户姓名等长期记忆，支持关键词召回和删除。
- `/chat` 接入会话历史加载、短期裁剪、用户记忆召回、会话保存和长期记忆提取。
- 新增 `GET /sessions/{session_id}` 查询会话状态。
- 新增 `DELETE /users/{user_id}/memories` 删除用户长期记忆。
- UI 新增用户记忆、会话摘要、保存偏好、召回记忆和清除记忆入口。
- 新增 M2 测试：短期记忆裁剪、SQLite 会话持久化、长期记忆保存/召回/删除、API 会话查询和跨会话记忆召回。

**验收结果**
- `pytest 05_PRODUCT_AGENT/tests -q`：22 passed。
- 100 轮消息可被裁剪为摘要 + 最近 8 轮，不发生上下文无限增长。
- 新会话可召回用户偏好，例如“我喜欢顺丰配送”。
- 删除用户记忆后，新会话不再召回被删除内容。
- 会话状态可通过 SQLite 重新实例化查询，满足本阶段恢复/查询要求。

**遗留到 M3**
- 当前记忆检索是轻量关键词匹配，不是向量检索；后续可替换为 Mem0 + pgvector。
- 当前没有 Redis 限流、Token 预算和模型 fallback，M3 需要补齐生产弹性。
- 当前质量分和 token 仍是轻量估算，M4 继续接入正式评估和指标。

---

### 2026-05-07：M3 限流与弹性完成

**实际交付**
- 新增 `RateLimiter`：支持用户每分钟请求限流、全局 QPS 限流、单次 Token 预算和全局小时 Token 预算。
- 限流采用 Hybrid 策略：未配置 `REDIS_URL` 时走内存计数，配置后走 Redis 计数；本地测试不需要外部服务。
- `/chat` 接入 M3 前置检查：用户限流、全局 QPS、上下文 token 估算和预算预留。
- Token 预算超限时返回简化回复，不执行客服图，并在响应中标记 `degraded` 和 `degrade_reason`。
- 新增 `ResilientLLM`：支持主模型重试、失败切备用模型、连续失败熔断后短期直走备用模型。
- UI 新增降级状态展示，`/health` 新增 `rate_limiter` 后端状态。
- 新增 M3 测试：用户第 11 次限流、QPS 超限、单次/全局预算降级、主备模型 fallback、重试和熔断。

**验收结果**
- `cd 05_PRODUCT_AGENT && pytest tests -q`：29 passed。
- 同一用户每分钟第 11 次请求返回 `429` 和友好中文提示。
- 全局 QPS 超限返回 `503` 和友好中文提示。
- Token 预算超限返回简化客服回复，服务不崩溃。
- `ResilientLLM` 在主模型失败后切换备用模型，连续失败达到阈值后熔断器开启。

**遗留到 M4**
- 当前 `ResilientLLM` 作为可测试弹性层存在，客服主路径仍是离线规则型；后续可按配置接入真实 LLM。
- Token 仍为轻量估算，M4 可结合真实模型 usage 和 Prometheus 指标做成本统计。
- Redis 集成未加入 Docker Compose；M5 部署阶段需要补齐 Redis 服务和压测配置。

---

### 2026-05-08：M4 可观测性与质量评估完成

**实际交付**
- 新增 `GET /metrics`：暴露 Prometheus 兼容文本指标，覆盖请求数、响应时间、Token 消耗、活跃会话、错误数、转人工数和质量分。
- 新增 LangSmith trace metadata 构建与环境配置：每次图调用携带 `session_id`、`user_id`、`environment`、`version`。
- 新增 `AutoQualityEvaluator`：按准确性、礼貌性、完整性计算加权质量分，并在低于阈值时记录告警事件。
- `/chat` 接入 M4 观测：成功、降级、限流和错误路径记录指标；会话 metadata 保存质量评估明细和告警标记。
- 新增 M4 测试：`/metrics` 指标出口、对话指标记录、低质量告警、trace metadata。

**验收结果**
- `cd 05_PRODUCT_AGENT && pytest tests/test_observability.py -q`：4 passed。
- `/metrics` 可被 Prometheus 按文本格式抓取。
- 质量分从占位值升级为确定性评估器输出。
- 低质量回答会产生 warning 级别结构化告警事件。

**遗留到 M5**
- 当前仅暴露 Grafana 所需指标，不提供 Grafana dashboard JSON 或容器编排；M5 部署阶段补齐。
- 当前质量评估器是离线确定性规则，真实 LLM-as-judge 或评估数据集将在 M6 强化。
- 当前 Token 用量仍为轻量估算，真实模型接入后需优先使用 provider usage 数据。

---

### 2026-05-08：M5 部署与压测完成

**实际交付**
- 新增 `Dockerfile`、`.dockerignore`、`docker-compose.yml`：支持本地一键启动 API、Redis、Postgres/pgvector、Prometheus、Grafana。
- 新增 `infra/prometheus.yml`：抓取 `api:8000/metrics`。
- 新增 Grafana provisioning：Prometheus datasource、dashboard provider 和客服 Agent 运营看板 JSON。
- 新增 `load_tests/locustfile.py`：覆盖订单、物流、商品、退款、转人工、健康检查和指标抓取场景。
- 更新 `.env.example`：移除真实密钥，补充 Compose、Grafana 和观测配置。
- 新增 M5 部署静态测试：校验 Docker、Compose、Prometheus、Grafana、Locust 和密钥防回归。

**验收结果**
- `cd 05_PRODUCT_AGENT && pytest tests/test_deployment.py -q`：6 passed。
- Compose 配置覆盖 M5 所需服务拓扑。
- Grafana dashboard 包含 QPS、平均响应时间、Token、错误率、转人工率和质量评分指标。

**遗留到 M6**
- 当前 Postgres/pgvector 作为部署拓扑服务启动，应用记忆实现仍使用 SQLite；后续可迁移到 pgvector/Mem0。
- 真实 24 小时长稳运行和正式压测报告待具备 Docker daemon/CI 环境后执行并归档。
- 真实 LLM 主路径、FAQ/RAG 工具和自动评测数据集仍在 M6 强化。

### 2026-05-08：M6 收尾强化完成

**实际交付**
- 新增 DeepSeek 真实 LLM 主路径：`LLM_MODE=deepseek` 时通过 `ChatOpenAI(base_url=https://api.deepseek.com)` 调用 `deepseek-v4-pro`，备用为 `deepseek-v4-flash`。
- 新增 FAQ/RAG 适配层：`rag/faq_tool.py` 尝试复用 `01_RAG` hybrid retriever，索引或依赖不可用时返回显式未命中，不影响 `/chat`。
- 新增管理接口：`GET /admin/sessions`、`GET /admin/users/{user_id}/memories`、`GET /admin/stats/transfers`。
- 新增 100 题评测集与自动评测报告：`evals/dataset.jsonl`、`evals/run.py`、`evals/report.py`。
- 更新 README 与 `.env.example`：补充 DeepSeek、管理接口、FAQ/RAG 和自动评测运行方式。

**验收结果**
- `cd 05_PRODUCT_AGENT && pytest tests/test_llm_factory.py tests/test_rag_faq.py tests/test_admin_api.py tests/test_evals.py -q`：15 passed。
- `cd 05_PRODUCT_AGENT && pytest tests -q`：59 passed。
- M6 评测数据集覆盖订单、物流、商品、退款、转人工、记忆、FAQ/RAG 和降级兜底场景。

**后续优化**
- 应用长期记忆仍使用 SQLite；Postgres/pgvector 与 Mem0 迁移可作为后续专项。
- 24 小时长稳运行、真实 Docker 压测报告和线上告警通道仍需在具备 Docker daemon/CI 环境后执行并归档。
- FAQ/RAG 当前通过适配层复用 01_RAG，本项目不复制 01 的索引构建流程。

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

- 长期记忆后续是否升级为 Mem0 + pgvector。
- 真实 24 小时长稳运行和正式 Docker 压测报告待执行归档。
- FAQ/RAG 的 01_RAG 索引构建仍由 01 项目负责，05 只维护适配层。
