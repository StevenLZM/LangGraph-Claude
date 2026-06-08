# 05_PRODUCT_AGENT 生产级 Agent 待优化项

更新时间：2026-06-08

本文档基于当前代码状态整理 05 项目距离“可承压上线的生产级 Agent”的主要差距。当前项目已经具备生产形态的教学骨架：FastAPI 入口、LangGraph 编排、三层记忆、检索决策、真实 LLM 主路径、幂等、会话锁、限流、Outbox、RocketMQ、Prometheus 指标、Grafana、Locust 和自动评测。后续优化重点不是再堆功能，而是把可靠性、安全边界、异步吞吐、记忆质量和运维闭环补实。

## 总体判断

当前 05 项目更接近“生产工程化教学样板”，还不是“可直接多实例、高并发、强合规上线”的生产系统。主要原因：

- `/chat` 是 async 入口，但核心路径仍混合同步 graph、同步存储、同步消息发送。
- Agent 形态是规则客服骨架加 LLM 最终回答，不是完整受控 tool-calling Agent。
- 订单、物流、商品、退款工具仍是 mock 数据。
- 三层记忆架构已经接入，但摘要、记忆抽取和 embedding 仍偏轻量。
- Admin、用户记忆删除、会话查询等接口还没有认证、授权和租户隔离。
- Outbox 和事件 handler 已有，但缺后台 worker、可靠重试、死信和重放闭环。

## P0 优先级

### 1. API 认证、租户隔离和 Admin 保护

当前依据：

- `api/routers/admin.py` 直接暴露 `/admin/sessions`、`/admin/users/{user_id}/memories`、`/admin/messages/outbox`。
- `api/main.py` 直接暴露 `GET /sessions/{session_id}` 和 `DELETE /users/{user_id}/memories`。
- `ChatRequest` 由客户端传入 `user_id` 和 `session_id`，当前没有从认证上下文派生身份。

生产风险：

- 任意调用方可以查看会话、读取或删除用户记忆。
- 多租户场景下，调用方可以伪造 `user_id` 访问其他用户上下文。
- 缺少审计日志，无法追踪敏感操作来源。

优化方向：

- 增加认证中间件或依赖注入，支持 JWT、API key 或内部服务 token。
- 从 token claims 派生 `tenant_id`、`user_id`、role，不信任请求体中的用户身份。
- Admin 路由增加 RBAC，只允许管理员或运营角色访问。
- 对记忆删除、会话查询、outbox 查询增加审计日志。
- 对响应中的长期记忆、会话内容和错误详情做 PII 脱敏。

验收标准：

- 未认证请求访问 `/chat`、`/sessions/*`、`/users/*/memories`、`/admin/*` 时被拒绝。
- 普通用户只能访问自己的 session 和 memory。
- Admin 操作写入审计事件，至少包含 actor、tenant、target、action、trace_id、时间。
- 测试覆盖跨用户访问被拒绝、Admin 权限成功、无权限删除失败。

### 2. 请求主路径去同步阻塞

当前依据：

- `api/main.py` 在 async `/chat` 中调用 `customer_service_graph.invoke(...)`。
- `memory/session_store.py`、`memory/long_term.py`、`memory/summary.py`、`api/idempotency.py` 使用同步 SQLite/Postgres 连接。
- `messaging/publisher.py` 在 `publish()` 中同步写 outbox 并同步发送 RocketMQ。
- `memory/semantic.py` 中 Milvus client 搜索和 upsert 也是同步调用。

生产风险：

- 高并发时事件循环会被同步 I/O 阻塞，吞吐和尾延迟恶化。
- Postgres 每次操作新建连接，没有连接池，连接成本和数据库压力偏高。
- RocketMQ 发送卡顿会拖慢用户可见响应。

优化方向：

- LangGraph 使用 `ainvoke`，节点内工具调用逐步改为 async。
- Postgres 存储改为 async 连接池，或将同步存储统一放入线程池并限制并发。
- RocketMQ 发送移出 `/chat` 主路径，请求内只写本地 outbox。
- Milvus 和 embedding 检索设置超时、并发限制和降级策略。
- 对每个外部依赖设置可配置 timeout。

验收标准：

- `/chat` 主路径没有未受控的同步网络 I/O。
- Postgres 后端复用连接池，压测时连接数稳定。
- RocketMQ 不可用时 `/chat` 不等待发送完成，只保留 pending outbox。
- Locust 或 pytest 并发用例覆盖不同 session 并发、同 session 串行、MQ 不可用降级。

### 3. 可靠后处理 worker 和 Outbox 重试闭环

当前依据：

- `messaging/handlers.py` 已有 `PostprocessEventHandler.handle()`。
- `messaging/outbox.py` 可以 enqueue、mark_published、mark_failed。
- `messaging/publisher.py` 发送失败时将状态标记为 `enqueue_failed`。
- 目前没有独立 consumer runner、outbox poller、死信队列或重放接口。

生产风险：

- 摘要记忆、语义记忆、质量评估等后处理不能保证最终执行。
- RocketMQ 短暂失败后没有自动补偿。
- handler 异常后缺少死信和人工重放入口。

优化方向：

- 增加 `workers/postprocess_consumer.py`，消费 `PostprocessRequested` FIFO 消息。
- 增加 outbox poller，扫描 pending/enqueue_failed 事件并按指数退避重试。
- 增加 dead-letter 状态和重放 API。
- 将后处理 handler 做成幂等消费：按 `event_id` 独立记录消费结果，而不是只写在 session metadata 中。
- 将 summary、semantic memory、quality metadata 更新放入清晰事务边界。

验收标准：

- RocketMQ 发送失败后事件留在 outbox，后台任务可重试并标记 published。
- Postprocess handler 重复消费同一 event_id 不重复写记忆。
- 失败超过阈值进入 dead-letter，可通过 Admin 重放。
- 测试覆盖发送失败、重试成功、重复消费、死信重放。

## P1 优先级

### 4. 从规则客服骨架升级为受控 tool-calling Agent

当前依据：

- `agent/graph.py` 已升级为多轮状态机主图，覆盖 `turn_router`、`slot_filling`、`confirmation_guard`、`tool_planner_or_react`、`tool_executor` 和 `response_builder`。
- `dialog_state` 已替代内部 `pending_choice`，退货/退款、物流追问和商品后续操作具备显式任务状态。
- 当前工具计划仍以规则和 mock 工具为主，受控 ReAct v1 仅覆盖只读工具循环。
- `api/main.py` 中真实 LLM 主要基于规则草稿和上下文生成最终回答。

生产风险：

- 更复杂的多意图、多工具、多轮澄清场景仍需要统一 tool schema 和更强评测覆盖。
- 规则分支容易和 LLM 最终回答产生语义偏差。
- 工具调用缺少统一 schema、预算、权限和错误恢复。

优化方向：

- 在现有受控状态机基础上继续完善 tool-calling schema：工具选择、参数生成、工具执行、工具结果校验、最终回答。
- 对工具定义统一 schema，包括输入、输出、权限、超时、幂等键和业务错误码。
- 增加澄清节点：缺订单号、缺确认、身份不匹配时先追问。
- 增加人工接管节点：高风险、投诉、法律、低置信度、工具失败时进入转人工。
- 保留规则 guardrail，作为 LLM 工具调用的安全边界。

验收标准：

- 订单、物流、退款、商品查询通过统一 tool schema 调用。
- LLM 不能越过后端工具结果编造订单、退款和库存。
- 工具失败时有明确用户响应和内部错误事件。
- 评测集覆盖多意图、缺参数、工具失败、人工接管。

### 5. 真实业务工具适配

当前依据：

- `agent/tools.py` 中 `MOCK_ORDERS`、`MOCK_LOGISTICS`、`MOCK_PRODUCTS` 是内存 mock。
- `apply_refund()` 只返回模拟退款提交结果。

生产风险：

- 业务数据不可用或不准确，无法真实服务用户。
- 退款等写操作缺少真实事务、审计和工单状态。
- 工具异常、慢调用、权限失败没有真实处理链路。

优化方向：

- 将工具层抽象为 adapter：OrderClient、LogisticsClient、CatalogClient、RefundClient。
- 每个 adapter 支持 timeout、retry、circuit breaker、业务错误映射。
- 退款写操作必须带确认状态、幂等键、审计事件和工单号。
- 增加 sandbox/mock 和 production 两种 provider，便于本地测试和真实部署切换。

验收标准：

- mock provider 和 production provider 共享同一接口。
- 退款确认二次提交可幂等，重复请求不重复创建工单。
- 工具超时、404、权限失败、业务拒绝都有稳定响应。
- 测试覆盖每个工具 adapter 的成功、失败、超时和幂等。

### 6. 三层记忆质量升级

当前依据：

- `memory/long_term.py` 的长期记忆抽取依赖关键词和正则。
- `memory/summary.py` 的 `build_conversation_summary()` 是最近消息拼接截断。
- `memory/semantic.py` 的 Milvus backend 使用 `_hash_embedding()` 生成哈希向量。
- `api/main.py` 会根据检索决策读取 keyword memory 和 semantic memory。

生产风险：

- 记忆抽取误判或漏判，容易保存无价值或敏感信息。
- 摘要不具备真正压缩、归纳和冲突处理能力。
- 哈希向量无法提供真实语义召回质量。
- 用户记忆缺少生命周期、置信度衰减和可解释治理。

优化方向：

- 接入真实 embedding 服务，并将 embedding 维度、模型版本、向量刷新策略配置化。
- 使用 LLM 进行摘要更新，保留事实、偏好、风险、待办和未解决问题。
- 长期记忆抽取使用结构化 LLM 输出，并进行敏感信息过滤。
- 增加记忆冲突合并策略，例如新的配送偏好覆盖旧偏好。
- 增加记忆可见、可编辑、可删除和审计能力。
- 建立记忆召回评测集，衡量 hit rate、precision、错误召回和延迟。

验收标准：

- Milvus 中记录真实 embedding、embedding_model、embedding_version。
- 摘要更新能保留关键事实并避免无限增长。
- 记忆抽取输出结构化 category/key/value/confidence/source。
- 用户可查看和删除长期记忆，删除同时清理 keyword 和 semantic backend。
- 评测报告包含记忆召回准确率和延迟。

### 7. 限流、预算和成本控制原子化

当前依据：

- `api/middleware/rate_limiter.py` 的用户限流和全局 QPS Redis 路径使用 Lua 计数。
- token budget Redis 路径使用 `get` 后再 `incrby`，不是原子检查加扣减。
- `memory/short_term.py` 和 graph 节点中的 token 数仍是轻量估算。

生产风险：

- 并发请求可能同时通过 token budget 检查，导致预算超扣。
- 估算 token 与真实模型 usage 偏差较大，成本控制不稳定。
- 缺少按租户、模型、渠道的成本维度。

优化方向：

- token budget 改为 Lua 原子脚本：检查剩余额度并一次性扣减。
- 接入 provider usage，响应后回写 prompt/completion/total token。
- 增加按 tenant、user、model、endpoint 的预算维度。
- 增加预算预留和实际消耗的差值校正。

验收标准：

- 并发 token budget 测试不会超扣。
- `/metrics` 暴露真实 token usage。
- 可按 tenant/model 查询预算消耗。
- 预算不足时返回稳定降级响应，并记录原因。

## P2 优先级

### 8. 健康检查和依赖就绪探测

当前依据：

- `api/main.py` 的 `/health` 返回配置状态和依赖配置情况，不真实 ping Redis、Postgres、RocketMQ、Milvus 或 LLM。

生产风险：

- 服务表面健康，但依赖不可用时仍接流量。
- 编排平台无法区分进程存活和业务就绪。

优化方向：

- 拆分 `/live` 和 `/ready`。
- `/live` 只检查进程存活。
- `/ready` 检查数据库、Redis、checkpointer、semantic memory、message outbox、LLM startup 状态。
- 依赖检查设置短 timeout，并返回脱敏错误。

验收标准：

- 依赖不可用时 `/ready` 返回非 200。
- `/live` 不因外部依赖短暂失败而失败。
- Docker Compose 或 K8s healthcheck 使用 `/ready`。

### 9. 观测体系增强

当前依据：

- `monitoring/metrics.py` 使用进程内数据结构渲染 Prometheus text。
- `monitoring/evaluator.py` 是规则评分，质量告警保存在进程内列表。
- `monitoring/tracing.py` 主要配置 LangSmith 环境和 trace metadata。

生产风险：

- 多 worker、多实例下指标不统一。
- 质量告警不能持久化，也无法形成运营闭环。
- trace 没有贯穿 LLM、工具、DB、MQ、memory retrieval 全链路。

优化方向：

- 使用 Prometheus client 原生 Counter、Histogram、Gauge。
- 增加结构化日志，字段包含 trace_id、tenant_id、user_id、session_id、request_id、tool_name。
- 为 LLM、工具、检索、MQ、DB 分别记录 latency、error、fallback、retry。
- 质量评估接入持久化告警表或事件流。
- 增加人工标注结果回流评测集。

验收标准：

- 指标包含 p50/p95/p99 延迟、工具失败率、LLM fallback 率、记忆召回命中率。
- 每次 `/chat` 可通过 trace_id 串起 API、graph、tool、LLM、outbox。
- 质量低分事件可在 Admin 或报告中查询。

### 10. 部署安全和生产运行配置

当前依据：

- `Dockerfile` 直接以 `uvicorn api.main:app` 启动单进程。
- `docker-compose.yml` 使用本地演示默认值，Grafana 默认密码可为 admin。
- 当前没有数据库迁移工具、K8s manifests、secret manager 或资源限制。

生产风险：

- 单进程承压能力有限。
- 配置和 secret 管理不适合真实环境。
- 自动建表和运行时 schema setup 可能影响启动稳定性。

优化方向：

- Dockerfile 使用非 root 用户，增加镜像安全扫描和依赖锁定。
- 使用 gunicorn + uvicorn worker 或明确的多副本部署策略。
- 增加 Alembic 或等价 migration 流程。
- 将 secret 放入外部 secret manager，不写入镜像和 repo。
- 增加 K8s deployment、service、configmap、secret、HPA、PDB、resource limit。
- 关闭生产环境自动 `checkpointer_setup`，由迁移任务预建表。

验收标准：

- 生产镜像非 root 运行。
- 数据库 schema 由 migration 管理。
- 生产配置不含默认密码。
- 部署文档包含滚动发布、回滚和扩缩容策略。

### 11. 评测和压测闭环

当前依据：

- `evals/dataset.jsonl` 已有 100 题自动评测。
- `load_tests/locustfile.py` 覆盖核心客服场景。
- 测试覆盖单元与接口行为，但真实依赖、真实 LLM 和多实例场景有限。

生产风险：

- 上线前无法量化真实 LLM、真实工具、真实记忆召回质量。
- 压测未覆盖同会话竞争、MQ 故障、Redis/Postgres 波动、LLM 慢调用。

优化方向：

- 增加分层评测：意图识别、工具调用、最终回答、记忆召回、人工转接。
- 增加真实依赖 smoke test 和 contract test。
- Locust 增加同 session 并发、重复 request_id、长对话、token 超限、MQ 失败场景。
- 输出发布前 readiness report。

验收标准：

- 每次主要变更生成 eval report 和 load test report。
- 关键场景通过率、p95 latency、错误率达到设定阈值。
- 失败用例能追溯到 trace_id 和具体模块。

## 建议迭代顺序

1. 先补认证、租户隔离、Admin 保护和审计日志。这是生产红线。
2. 再补 postprocess worker、outbox 重试、死信和重放，让三层记忆真正异步可靠。
3. 然后改 async store、连接池、graph `ainvoke` 和 RocketMQ 主路径解耦，解决并发吞吐。
4. 接真实 embedding、LLM 摘要和结构化记忆治理，提升记忆质量。
5. 接真实业务工具 adapter，并补工具权限、幂等和工单审计。
6. 最后升级健康检查、观测、部署、压测和评测闭环。

## 后续拆分建议

这些优化不建议一次性实现。建议拆成五个专项：

- 安全专项：认证、租户、RBAC、审计、PII 脱敏。
- 可靠异步专项：async store、postprocess worker、outbox retry、DLQ。
- 记忆质量专项：真实 embedding、LLM 摘要、结构化长期记忆、召回评测。
- 工具生产化专项：真实业务 adapter、tool schema、幂等和错误恢复。
- 运维专项：ready check、Prometheus client、结构化日志、K8s、migration、压测报告。
