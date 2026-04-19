# InsightLoop（03_MULTI_AGENT）

AI 行业深度研究多 Agent 系统。设计文档：
- [PROJECT_SCENARIO.md](./PROJECT_SCENARIO.md) — 产品定位 / Agent 分工 / 验收标准
- [ENGINEERING.md](./ENGINEERING.md) — 工程设计：架构、State、HITL、并行、MCP 双向接入

## 当前进度

**M2-M4 后端闭环 + Internal/External MCP 集成**：

- ✅ Planner: qwen-max 拆解子问题 + `interrupt()` HITL
- ✅ Supervisor: 条件路由 + Send fan-out
- ✅ 4×Researcher 并行: Tavily / ArXiv / GitHub / KB（复用 01_RAG 混合检索）
- ✅ Reflector: LLM 覆盖度评分 + 补查/收敛 + 3 轮硬兜底
- ✅ Writer: qwen-max Markdown 报告 + [^N] 引用脚注 + 落盘归档
- ✅ Evidence reducer: URL 去重 + relevance_score 排序
- ✅ FastAPI: `/research` `/resume` `/turn` `/state` `/threads` `/reports`
- ✅ **外部 MCP Client**：官方 `mcp` SDK + Brave Search MCP（降级链位置 2）
- ✅ **内部 MCP Server**：5 个 tool 暴露本项目能力给 Claude Desktop
- ✅ 多轮会话: AsyncSqliteSaver checkpointer
- ✅ Web 降级链：`Tavily → Brave MCP → DashScope 内置搜索 → 跳过`

## 环境要求

**Python 3.11+**（官方 MCP SDK 要求 ≥ 3.10；langgraph async interrupt 在 3.11 才稳）。

```bash
conda create -n langgraph-cc-multiagent python=3.11 -y
conda activate langgraph-cc-multiagent
cd 03_MULTI_AGENT
pip install -r requirements.txt
cp .env.example .env  # 填入真实 key
```

**Node.js ≥ 18**（`npx` 启动 MCP server 需要）。

## 启动

### 1. 测试套件（14 单测，离线）

```bash
PYTHONPATH=. pytest tests/ -v
```

### 2. 本地真实跑（CLI）

```bash
PYTHONPATH=. python -m scripts.run_local "对比 LangGraph 与 AutoGen 的核心抽象差异"
```

### 3. 单独验证 Brave MCP 能否通（smoke）

```bash
PYTHONPATH=. python -m scripts.test_brave_mcp "LangGraph 2025"
```

> ⚠️ `api.search.brave.com` 在国内网络可能超时。Brave MCP 失败时 registry 自动降级到 DashScope 内置搜索（零 key，国内可用）。

### 4. 启动 API

```bash
PYTHONPATH=. uvicorn app.api:app --reload --port 8080
```

```bash
# 启动研究（在 planner 处 interrupt）
curl -X POST localhost:8080/research -H 'Content-Type: application/json' \
  -d '{"research_query":"分析 2025 年开源 Agent 框架格局"}'
# → {"thread_id":"xxxx","interrupt":{"plan":{"sub_questions":[...]}}}

# 用户编辑后 resume
curl -X POST localhost:8080/research/xxxx/resume -H 'Content-Type: application/json' \
  -d '{"plan":{"sub_questions":[...],"estimated_depth":"standard"}}'

# 同会话追问
curl -X POST localhost:8080/research/xxxx/turn -H 'Content-Type: application/json' \
  -d '{"research_query":"再深入对比 state 管理"}'
```

### 5. 内部 MCP server（供 Claude Desktop / Cursor 用）

```bash
PYTHONPATH=. python -m tools.internal_mcp.server  # stdio 模式
```

Claude Desktop `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "insightloop": {
      "command": "python",
      "args": ["-m", "tools.internal_mcp.server"],
      "cwd": "/绝对路径/03_MULTI_AGENT",
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

## 架构亮点

1. **Supervisor + Send fan-out**：1 次决策派发 N 个并行 Researcher（4 源 × M 子问题）；`merge_evidence` reducer 自动 URL 去重 + relevance 排序
2. **`interrupt()` + `Command(resume=...)` HITL**：Planner 暂停等用户编辑计划；同 thread_id 跨 turn 状态恢复
3. **Reflector 反思循环**：LLM 评分 evidence 覆盖度 → 决定补查/收敛；`max_iterations=3` 硬兜底
4. **统一 SearchTool 协议 + ToolRegistry 降级链**：HTTP 工具、官方 MCP 工具、LLM 内置搜索**同一协议**，Researcher 无 if/else
5. **MCP 双向**：
   - **Client**：官方 `mcp.ClientSession` 连接 Brave MCP（stdio 子进程，lifespan 绑定）
   - **Server**：暴露 `kb_search` / `list_reports` / `read_report` / `list_evidence` / `trigger_research` 5 个 tool 给 Claude Desktop
6. **复用 01_RAG**：`tools/kb_retriever.py` 通过 surgical sys.path/sys.modules 隔离，在不修改 01_RAG、不污染本项目 import 解析的前提下加载其 `ParentChildHybridRetriever`
7. **纯 async 图**：Py3.11+ 下所有节点 async，`await graph.ainvoke` 并发调度 HTTP/LLM，AsyncSqliteSaver 异步持久化

## 架构图

```
              ┌─────────────────────────────┐
    HITL      │  Planner (qwen-max)         │
   ◀──────────│  + interrupt({plan})        │
   resume     └──────────────┬──────────────┘
                             │
                  ┌──────────▼──────────┐
                  │   Supervisor        │
                  │  (Send fan-out)     │
                  └──────────┬──────────┘
             ┌───────────────┼───────────────┬─────────────┐
             ▼               ▼               ▼             ▼
      ┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐
      │   Web    │    │ Academic │    │   Code   │   │    KB    │
      │ Tavily + │    │  ArXiv   │    │  GitHub  │   │ 01_RAG   │
      │ Brave MCP│    │          │    │          │   │ hybrid   │
      │+DashScope│    │          │    │          │   │          │
      └────┬─────┘    └────┬─────┘    └────┬─────┘   └────┬─────┘
           └───────────────┼───────────────┴───────────────┘
                           ▼
                  ┌──────────────────┐
                  │    Reflector     │◀──┐
                  │  (cov/补查/兜底) │   │
                  └────────┬─────────┘   │ need_more
                           │ sufficient  │
                           ▼             │
                  ┌──────────────────┐   │
                  │     Writer       │   │
                  │ (qwen-max md)    │   │
                  │  → data/reports/ │   │
                  └────────┬─────────┘   │
                           ▼             │
                         [END]───────────┘
```

## 后续里程碑

- **M5**: FastAPI SSE 流式推送 + Streamlit UI
- **M6**: 20 题评测集 + LLM-as-judge + Docker Compose
