# AI Agent 工程师实战项目路线图

> 目标：通过 5 个由浅入深的实战项目，系统掌握 AI Agent 技术栈，达到应聘 AI Agent 工程师的标准。

---

## 项目总览

| # | 项目名称 | 核心技术 | 难度 | 周期 |
|---|----------|----------|------|------|
| 01 | [RAG 知识库问答系统](./01_rag_knowledge_base.md) | LangChain · Chroma · Embedding | ⭐⭐ | 2 周 |
| 02 | [ReAct Agent 智能工具助手](./02_react_agent_tools.md) | Function Calling · AgentExecutor · 自定义工具 | ⭐⭐⭐ | 2 周 |
| 03 | [多 Agent 协作内容创作团队](./03_multi_agent_content_team.md) | LangGraph · StateGraph · 角色编排 | ⭐⭐⭐⭐ | 3 周 |
| 04 | [Human-in-the-Loop 合同审核](./04_human_in_the_loop_contract.md) | LangGraph Interrupt · Checkpoint · 状态恢复 | ⭐⭐⭐⭐ | 3 周 |
| 05 | [生产级智能客服平台](./05_production_agent_customer_service.md) | 监控 · 限流 · 记忆管理 · 容器化 | ⭐⭐⭐⭐⭐ | 4 周 |

**总计：约 14 周（3.5 个月）**

---

## 技术栈覆盖全景

```
项目01  ████████░░░░░░░░  RAG / 向量数据库 / Embedding
项目02  ██████████░░░░░░  工具调用 / ReAct / Function Calling
项目03  ████████████░░░░  LangGraph / 多 Agent / 状态管理
项目04  ██████████████░░  HITL / Checkpoint / 持久化
项目05  ████████████████  生产级：监控/限流/记忆/容器化/评估
```

---

## 每个项目完成后的能力验收

### 项目01 完成后，你能：
- 独立搭建 RAG 系统，解决私有知识库问答场景
- 解释 Embedding、向量检索、相似度匹配的原理
- 使用 LangChain LCEL 构建检索链

### 项目02 完成后，你能：
- 设计和封装自定义工具（符合 Function Calling 协议）
- 理解 ReAct 推理循环，能调试 Agent 的思考过程
- 处理工具调用失败的错误恢复

### 项目03 完成后，你能：
- 使用 LangGraph 构建有状态的多 Agent 工作流
- 设计 Agent 之间的信息传递协议
- 实现条件路由和循环控制逻辑

### 项目04 完成后，你能：
- 实现 Agent 执行的暂停和恢复（interrupt/resume）
- 设计需要人工介入的 AI 系统
- 实现状态持久化，保证系统故障可恢复

### 项目05 完成后，你能：
- 将 Agent 系统从 Demo 推向生产级别
- 设计完整的监控、告警、限流方案
- 管理 LLM 的 Token 成本，设计降级策略
- 对 Agent 回答质量进行自动化评估

---

## 面试准备检查清单

完成所有项目后，确认以下能力：

### 技术深度
- [ ] 能解释 RAG 和 Fine-tuning 的区别及适用场景
- [ ] 能画出 LangGraph 的执行模型（节点/边/状态）
- [ ] 能解释 ReAct 和 Plan-and-Execute 的区别
- [ ] 了解 Agent 记忆的四种类型（感知/短期/长期/外部）

### 工程实践
- [ ] 至少 1 个项目有完整的 GitHub README + 演示截图/视频
- [ ] 能描述在生产中遇到的问题及解决方案
- [ ] 了解 LLM 幻觉问题及缓解手段
- [ ] 熟悉 LangSmith 等 LLMOps 工具

### 业务理解
- [ ] 能针对真实业务场景设计 Agent 架构
- [ ] 了解 Token 成本控制的常见方法
- [ ] 了解 AI 系统的安全风险（Prompt Injection 等）

---

## 推荐学习资源

| 资源 | 类型 | 说明 |
|------|------|------|
| LangChain 官方文档 | 文档 | 框架基础，必读 |
| LangGraph 官方文档 | 文档 | 图编排核心，重点学 |
| LangSmith 文档 | 文档 | 调试和监控必备 |
| DeepLearning.AI 短课 | 视频 | LangChain/Agent 系列 |
| LangChain GitHub | 代码 | 大量示例可参考 |
