# 工程设计总览

> 5 个实战项目的完整工程设计文档索引

---

## 文档列表

| # | 工程设计文档 | 核心技术点 |
|---|------------|-----------|
| 01 | [RAG 知识库问答](./01_rag_engineering.md) | LCEL · 混合检索(RRF) · 语义分块 · MCP Filesystem |
| 02 | [ReAct 工具调用 Agent](./02_react_agent_engineering.md) | LangGraph ToolNode · 自定义工具 · MCP Server实现 · 流式输出 |
| 03 | [多 Agent 协作](./03_multi_agent_engineering.md) | StateGraph · 条件路由 · 修改循环 · LangSmith Tracing |
| 04 | [Human-in-the-Loop](./04_hitl_engineering.md) | interrupt() · SqliteSaver · FastAPI恢复接口 · 审计日志 |
| 05 | [生产级客服平台](./05_production_engineering.md) | 短/长期记忆 · 限流熔断 · Prometheus · Docker Compose |

---

## 技术栈全景（按项目累积）

```
框架      LangChain LCEL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          LangGraph      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
检索      Chroma向量库   ━━
          混合检索(RRF)  ━━
工具      Function Call  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          MCP Server     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
多Agent   StateGraph     ━━━━━━━━━━━━━━━━━━━━━━━━━━━
          条件路由       ━━━━━━━━━━━━━━━━━━━━━━━━━━━
HITL      interrupt()    ━━━━━━━━━━━━━
          Checkpoint     ━━━━━━━━━━━━━
记忆      短期/Token管理 ━━
          长期/Mem0      ━━
生产      限流/熔断      ━━
          监控/LangSmith ━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Docker部署     ━━
          项目  01  02  03  04  05
```
