# 开发进度日志（DEV_PROGRESS）

> 本文档是 Claude 跨会话的工程记忆 —— 读完本文即可掌握完整开发脉络、已做决策、当前状态、未竟事项。
>
> 最后更新：2026-04-26
> 当前阶段：**M5 完成 + M6 部分完成**（LangSmith 自动追踪 + LLM-as-judge + 评测看板）；M6 剩 Docker 与 20 题完整集

---

## 一、项目速查

| 维度 | 内容 |
|---|---|
| 名称 | InsightLoop —— AI 行业深度研究多 Agent 系统 |
| 定位 | 生产级 multi-agent，供 AI Agent 工程师面试作品集使用 |
| PRD | `PROJECT_SCENARIO.md` |
| 工程设计 | `ENGINEERING.md` |
| 关键依赖 | LangGraph 1.1+ · LangChain · ChatOpenAI(DashScope compat) · 官方 mcp SDK · FastAPI · AsyncSqliteSaver · httpx |
| Python | **3.11.15** （conda env `langgraph-cc-multiagent`） |
| Node.js | **22.x** （用于 npx 启动 MCP server） |

启动 3 行：
```bash
conda activate langgraph-cc-multiagent
cd 03_MULTI_AGENT
PYTHONPATH=. python -m scripts.run_local "研究问题"
```

---

## 二、开发里程碑时间线

### M1 骨架（已完成）
- 建立全目录结构（app/graph/agents/tools/prompts/config/tests 等 ~40 文件）
- State / SubQuestion / Evidence / ResearchPlan / ReflectionResult / Citation 契约
- safe_node 装饰器、ToolRegistry、SearchTool Protocol、merge_evidence reducer
- 7 个 agent 节点空桩 + 主图装配（planner → supervisor → fan-out → 4 researcher → reflector → writer）
- 空 FastAPI + `pytest` 通过

### M2 HITL + 真实 LLM（已完成）
- `config/llm.py`：ChatOpenAI 走 DashScope OpenAI 兼容端点，tier=max/turbo
- `agents/planner.py`：真实 LLM + `interrupt()` + 用户编辑计划后 `Command(resume=...)` 恢复
- 结构化输出：`llm.with_structured_output(Model, method="function_calling")` —— DashScope compat 端点必须用 function_calling，否则会报 "'messages' must contain 'json'"

### M3 并行 + 反思（已完成）
- `agents/_researcher_base.py`：统一 `run_research_chain` 从 registry 拿降级链，顺序尝试，首个非空结果即返回
- 4 个 Researcher 节点（web/academic/code/kb）通过 registry 调工具
- `agents/reflector.py`：LLM 对 evidence 覆盖度打分；`next_action ∈ {sufficient, need_more_research, force_complete}`；第 3 轮硬兜底不调 LLM
- Evidence 聚合：`merge_evidence` reducer 按 URL 去重、relevance_score 倒序

### M4 报告 + 归档（已完成）
- `agents/writer.py`：LLM 生成 Markdown（含 `[^N]` 引用），后处理补全引用章节
- `app/report_store.py`：按 `{ts}_{slug}_{tid}.md` 归档，支持 `find_by_thread`
- FastAPI 端点：`/research` `/resume` `/turn` `/state` `/threads` `/reports`

### MCP 双向集成（已完成）
- **External client**（`tools/mcp_brave_tool.py`）：官方 `mcp.ClientSession + stdio_client + AsyncExitStack`；连接 `npx -y @modelcontextprotocol/server-brave-search`
- **Internal server**（`tools/internal_mcp/`）：暴露 5 个 tool 给 Claude Desktop
  - `kb_search` / `list_reports` / `read_report` / `list_evidence` / `trigger_research`
  - handlers 全部真实实现（非 stub）
  - 支持 stdio 启动：`python -m tools.internal_mcp.server`

### DeepSeek 切换（2026-04-25）
- `config/llm.py`：LLM 工厂从 DashScope 切到 DeepSeek（`deepseek-v4-pro` / `deepseek-v4-flash`），走 OpenAI 兼容端点 `https://api.deepseek.com`
- `agents/planner.py`、`agents/reflector.py`：`with_structured_output(..., method="json_mode")`（DashScope 时代用的 `function_calling` 不再适用，DeepSeek 原生支持 JSON mode）
- DashScope 留作搜索兜底（`tools/dashscope_search_tool.py`），LLM 通道完全切走
- 新增 `tests/test_deepseek_structured_output.py` 锁住 method 不被回退

### M5 SSE + Streamlit UI（已完成，2026-04-26）
- `app/sse.py`：纯函数把 LangGraph `astream_events(v2)` 事件 → SSE event dict
  - `on_chain_start/_end` 仅过 8 个业务节点；非业务 run（ChannelWrite 等）丢弃
  - `on_chat_model_stream` **只对 `langgraph_node == "writer"` 转 token**，避免 planner/reflector 中间 token 轰炸
  - 顶层图 `LangGraph` 的 on_chain_end 输出含 `__interrupt__` 时发 `interrupt` 事件并跳过 `done`
  - `coerce_plan_payload` 支持 dict / JSON str / `{"plan": {...}}` 三形态
- `app/api.py` 新增 3 条 SSE 端点（旧端点全部保留）：
  - `GET /research/stream?query=...&audience=...` 启动并流推
  - `GET /research/{tid}/resume_stream?plan=<json>` 恢复 interrupt
  - `GET /research/{tid}/turn_stream?query=...` 同会话追问
  - 用 `sse_starlette.EventSourceResponse`，`ping=15`、`send_timeout` 来自 `settings.sse_retry_ms`
- `app/streamlit_ui.py`：单页聊天 UI
  - 8:4 双栏；左侧报告区（writer token 流式 markdown），右侧 8 节点状态指示（⚪️/⏳/✅）+ 最近 5 条工具调用
  - 命中 `interrupt` 时弹出 `st.expander` + `st.data_editor` 让用户改 sub_questions
  - 报告完成后底部启用追问输入框，复用 thread_id 走 `turn_stream`
  - 用 `httpx-sse.connect_sse` 同步消费，每 12 个 token 刷一次 placeholder（避免 Streamlit rerun 风暴）
- 测试：`tests/test_sse_stream.py` 8 用例（事件映射规则 + ASGITransport 端点冒烟），全量 49 用例绿

### 已完成的配套
- **Python 3.11 升级**：conda env `langgraph-cc-multiagent`；langgraph 1.1.6、mcp 1.27.0、langchain 1.2.15 等
- **requirements.txt** 对齐 3.11 生态
- **DashScope 内置搜索兜底**（`tools/dashscope_search_tool.py`）：直调 DashScope 原生端点 `/api/v1/.../generation`，读 `output.search_info.search_results`；用作国内可达的 web 兜底
- **01_RAG 复用**：`tools/kb_retriever.py` 用 surgical sys.path/sys.modules 隔离加载
- **49 单测全绿**（M5 加 8 条 SSE 用例、tutorial 子目录 +N 条）

### M6 LangSmith + LLM-as-judge（已完成，2026-04-26）
- **LangSmith 自动追踪**（`app/bootstrap.py:_setup_langsmith`）：检测到 `langchain_tracing_v2=true` + key 时把 `LANGCHAIN_*` 同步到 `os.environ`，LangChain 全局 callback 自动上报；不改任何节点代码
  - `app/api.py:_config(tid, query, audience)` 给 RunnableConfig 注入 `metadata={thread_id, research_query, audience, app}`，云端可按 thread / 问题筛 trace
  - 评测处再加 `metadata.eval_run_id / case_id` 与 `tags=["eval"]`，把评测 trace 与 demo trace 区分开
  - `config/tracing.py:with_tags` 仍是兼容钩子（节点级硬 tag 留作后续优化）
- **LLM-as-judge**（`evals/judge.py`）：DeepSeek pro `with_structured_output(JudgeScore, method="json_mode")` 三维度打分
  - 维度：覆盖度 / 准确性 / 引用质量；overall = 0.4·cov + 0.3·acc + 0.3·cit
  - prompt 同时塞 query / plan / evidence_brief / report_md；报告超 6000 字截断
- **评测脚本**（`evals/run.py`）：`PYTHONPATH=. python -m evals.run [--limit N]`
  - 每题 graph.ainvoke → interrupt → 自动 accept LLM plan → resume → judge；失败不中断整轮
  - 产出 `evals/results/{run_id}/results.jsonl` + `REPORT.md`
- **Markdown 报告**（`evals/report.py`）：表格 + 维度均值 + 失分案例（综合 < 70 或执行失败）
- **Streamlit 看板**（`app/evals_ui.py`）：`streamlit run app/evals_ui.py`
  - 单 run：4 项指标卡 + 维度均值柱状图 + 用例 dataframe + 单条详情（rationale + 报告全文 expander）
  - 两 run 对比：维度均值并排柱状图 + 逐用例 overall delta 表
- **数据集**（`evals/dataset.jsonl`）：5 题烟测（技术 / 产业 / 对比 / 追问），后续可扩到 20
- **53 单测全绿**（M6 加 4 条 evals 用例）

### 待启动里程碑
- **M6 剩余**：20 题完整数据集；Dockerfile + docker-compose 一键启动；`config/tracing.py` 节点级手动 tag

---

## 三、关键架构决策及其由来（踩坑备忘）

### 1. Python 3.9 → 3.11 的必要性
- **MCP 官方 SDK 需要 Python ≥ 3.10**，3.9 装不上
- **`langgraph.types.interrupt()` 在 Python 3.9 + 异步节点里 contextvars 丢失**，langgraph 源码里会显式提示 "Python 3.11 or later required to use this in an async context"
- 3.11 之前的 workaround：把节点写成 sync + FastAPI 用 `asyncio.to_thread(graph.invoke, ...)`；但这样需要 LLM 也走 sync invoke，对长输出（Writer qwen-max 写 3000+ 字）稳定性差；且 mcp SDK 仍然装不上
- **结论**：必须升 3.11。所有节点已恢复纯 async，API 用 `await graph.ainvoke()`

### 2. DashScope OpenAI 兼容端点 vs 原生端点
- **LLM 调用走 compat 端点**（`https://dashscope.aliyuncs.com/compatible-mode/v1`）：因为 LangChain `ChatOpenAI` 生态成熟、`with_structured_output` 只支持 OpenAI 协议
- **但 `with_structured_output` 必须传 `method="function_calling"`**；默认 `json_object` 模式 DashScope 会要求 prompt 里包含 "json" 字样，会挂
- **DashScope 内置联网搜索走原生端点**（`https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`）：因为 OpenAI compat 端点会丢掉 `search_info` 非标字段；原生端点才返回结构化 `search_results`

### 3. Evidence reducer 为什么自定义而不是 `operator.add`
- `operator.add` 简单拼接会留重复 URL
- `merge_evidence(old, new)`：按 URL 聚合，同 URL 保留 relevance_score 最高的一条；输出按 score 倒序
- 兼容 Pydantic 对象和 dict 两种输入（LangGraph 在序列化/反序列化时可能把 Evidence 转为 dict）

### 4. 01_RAG 复用的 import 冲突
- 01_RAG 目录下有自己的 `mcp/`、`config/` 子包，会与本项目同名模块、官方 MCP SDK 冲突
- `tools/kb_retriever.py` 用 **surgical sys.path/sys.modules 隔离**：
  1. 备份本项目 `config` 和 `mcp` 模块对象
  2. 临时把 01_RAG 路径插到 sys.path 最前
  3. 从 sys.modules 弹出 `config`，让 `from rag.retriever import ...` 触发时解析到 01_RAG/config
  4. finally 块里恢复 sys.path 与 sys.modules，清理 01_RAG 注入的副作用
- rag.retriever 加载完后，其内部引用的 `config` 已绑定到 01_RAG/config 模块对象（Python import 机制缓存），独立可用
- **关键**：不修改 01_RAG 任何文件；不把 01_RAG 加到全局 PYTHONPATH（否则其 `mcp/` 子包遮蔽官方 SDK）

### 5. Web 降级链设计
顺序：`Tavily → Brave MCP → DashScope 内置搜索 → 跳过`
- Tavily 主源（需 `TAVILY_API_KEY`）
- Brave MCP：通过官方 mcp SDK 连 `@modelcontextprotocol/server-brave-search`；国内直连超时（api.search.brave.com 不通），会返空自动降级
- DashScope 内置搜索：零额外 key，国内可达，最终兜底
- 工具返空 → registry 自动下一个；全空 → 该 source_type 该轮无 evidence，reflector 会看到缺口并决定补查

### 6. 外部 MCP 代理问题（已放弃）
- 国内访问 `api.search.brave.com` 需代理（`127.0.0.1:7890` via Clash 等）
- Node 22 原生 fetch **不读 `HTTPS_PROXY` 环境变量**
- 需用 `undici.ProxyAgent` + `NODE_OPTIONS=--import` 注入 ESM bootstrap 脚本才行
- **用户选择不接代理**，当前状态：Brave MCP 代码完整、协议握手通，只是运行时 fetch 永远失败 → 降级链自动兜底
- 保留代码级 hook：`settings.https_proxy`、`MCPBraveSearchTool(proxy=...)` —— 将来海外部署 / 换网络时一行 env 就能启用

### 7. AsyncSqliteSaver
- 初次实现用 `SqliteSaver`（sync），后来换成 `ainvoke` 时报 "SqliteSaver does not support async methods"
- 改 `AsyncSqliteSaver`（需 `aiosqlite`）；它是 async context manager，在 `bootstrap.startup()` 里通过 `AsyncExitStack` 管理生命周期，`shutdown()` 时 `aclose`

---

## 四、代码组织现状

```
03_MULTI_AGENT/
├── PROJECT_SCENARIO.md              # PRD，不改动
├── ENGINEERING.md                   # 工程设计，可增量更新
├── DEV_PROGRESS.md                  # 本文件
├── README.md                        # 启动说明
│
├── app/
│   ├── api.py                       # FastAPI 路由（async）
│   ├── bootstrap.py                 # 启动/关闭；AsyncSqliteSaver + AsyncExitStack
│   ├── schemas.py                   # StartReq/ResumeReq/TurnReq/StartResp
│   ├── turn_init.py                 # reset_per_turn 多轮状态清理
│   └── report_store.py              # 报告归档 + list/read/find_by_thread
│
├── graph/
│   ├── state.py                     # ResearchState + merge_evidence reducer
│   ├── workflow.py                  # build_graph（纯 async）
│   ├── router.py                    # supervisor_route + reflector_route
│   └── nodes_parallel.py            # fanout 导出（实际逻辑在 router.supervisor_route 里返 list[Send]）
│
├── agents/
│   ├── schemas.py                   # Pydantic 契约
│   ├── _safe.py                     # safe_node 装饰器
│   ├── _researcher_base.py          # run_research_chain + extract_sq_and_query
│   ├── planner.py                   # LLM + interrupt()
│   ├── supervisor.py                # 简单状态标记 + iteration++
│   ├── researcher_{web,academic,code,kb}.py
│   ├── reflector.py                 # LLM 覆盖度评分，rc>=3 硬兜底
│   └── writer.py                    # LLM Markdown + 引用列表 + 落盘
│
├── tools/
│   ├── base.py                      # SearchTool Protocol + ToolResult
│   ├── registry.py                  # ToolRegistry
│   ├── _http.py                     # httpx AsyncClient + tenacity 重试
│   ├── tavily_tool.py               # POST Tavily /search
│   ├── arxiv_tool.py                # ArXiv Atom XML 解析
│   ├── github_tool.py               # GitHub Search API
│   ├── dashscope_search_tool.py     # DashScope 原生端点 + search_info
│   ├── kb_retriever.py              # 01_RAG surgical 隔离加载
│   ├── mcp_loader.py                # 解析 .mcp.json + 注册 brave-search
│   ├── mcp_brave_tool.py            # 官方 mcp SDK ClientSession
│   └── internal_mcp/
│       ├── server.py                # 5 tool 暴露
│       ├── schemas.py               # inputSchema
│       └── handlers.py              # 真实实现
│
├── prompts/templates.py             # 各 agent prompt
├── config/
│   ├── settings.py                  # pydantic-settings
│   ├── llm.py                       # get_llm(tier) ChatOpenAI 工厂
│   └── tracing.py                   # with_tags（骨架，M6 接 LangSmith）
│
├── scripts/
│   ├── run_local.py                 # CLI 端到端跑（自动接受 plan）
│   └── test_brave_mcp.py            # Brave smoke 脚本
│
├── tests/                           # 14 单测全绿
│   ├── test_graph_skeleton.py
│   ├── test_evidence_reducer.py
│   ├── test_registry.py
│   ├── test_dashscope_search.py
│   ├── test_mcp_brave.py
│   └── test_end_to_end_offline.py   # mock LLM + mock web tool 的完整 HITL 流
│
├── data/
│   ├── documents/                   # 可放 PDF 供 KB（复用 01_RAG 索引时此处空即可）
│   └── reports/                     # Writer 输出归档
│
├── .env                             # 真实 key（允许提交，私有项目）
├── .env.example                     # 示例
├── .mcp.json                        # 外部 MCP server 配置
└── requirements.txt
```

**禁区**：不修改 `01_RAG/` 任何文件。

---

## 五、运行方式

### 测试（离线，不碰外部 API）
```bash
PYTHONPATH=. pytest tests/ -v     # 14 passed
```

### 真实端到端（CLI）
```bash
PYTHONPATH=. python -m scripts.run_local "对比 LangGraph 与 AutoGen 的核心抽象差异"
```
行为：Planner LLM 拆 5 个子问题 → 自动接受 → 并行 4 researchers → Reflector → Writer 3000+ 字 Markdown → 落盘 `data/reports/`

### FastAPI
```bash
PYTHONPATH=. uvicorn app.api:app --reload --port 8080
```
```bash
curl -X POST localhost:8080/research -H 'Content-Type: application/json' \
  -d '{"research_query":"..."}'
# 返 interrupt；编辑后
curl -X POST localhost:8080/research/{tid}/resume -H 'Content-Type: application/json' \
  -d '{"plan": {...}}'
```

### Internal MCP（供 Claude Desktop）
```bash
PYTHONPATH=. python -m tools.internal_mcp.server
```

### Brave MCP smoke（海外 / 代理环境下验证）
```bash
PYTHONPATH=. python -m scripts.test_brave_mcp "LangGraph 2025"
```

---

## 六、已知约束

| 现象 | 原因 | 应对 |
|---|---|---|
| Brave MCP 国内返空 | `api.search.brave.com` 不直连，Node fetch 不读 HTTPS_PROXY | 降级链自动落 DashScope；海外部署或自行接代理时可用 |
| Tavily 配额限制 | 免费档每分钟 10 次 | registry 降级链自动后退 |
| Writer 报告质量依赖 evidence | evidence 少时容易引用不足 / 编造 | Reflector 覆盖度评分 + 第 2 轮补查；Writer prompt 明确要求"不编造" |
| KB Researcher 需 01_RAG 索引存在 | `ParentChildHybridRetriever.invoke` 需向量库+BM25 就绪 | 01_RAG 未就绪时 `KBRetriever.search()` 返空，不影响其他源 |
| `pytest-asyncio` 在 3.11 warnings | 节点函数签名中 `RunnableConfig | None` 在 3.11 之前类型解析提示 | 已用 `Optional[RunnableConfig]` 修掉；剩余 2 个 warning 可忽略 |

---

## 七、key 现状（.env 实际值，2026-04-18）

| 变量 | 状态 | 备注 |
|---|---|---|
| DEEPSEEK_API_KEY | ✅ 已填 | deepseek-v4-pro / deepseek-v4-flash（M4 末换掉 DashScope LLM） |
| DASHSCOPE_API_KEY | ✅ 已填 | 仅用于内置搜索兜底（LLM 通道已切走） |
| TAVILY_API_KEY | ✅ 已填 | web 主源 |
| GITHUB_TOKEN | ✅ 已填 | code 源 |
| BRAVE_API_KEY | ✅ 已填 | 但国内不可达 |
| LANGCHAIN_API_KEY | ✅ 已填 | LangSmith，M6 启用 |
| HTTPS_PROXY / HTTP_PROXY | ❌ 留空 | 用户明确不接代理 |

---

## 八、下次开发建议的起点

### 优先级 A：M6 收尾（面试亮点剩余）
1. `evals/dataset.jsonl` 扩到 20 题（5×{技术、产业、对比、追问} × audience 变化）
2. `Dockerfile` + `docker-compose.yml`：app + chroma 两 service（按用户偏好仅打包 03_MULTI_AGENT）
3. 可选：`config/tracing.py:with_tags` 真实接入 RunnableConfig.merge —— 节点级 tag 让 LangSmith filter 更精细

### 优先级 B：锦上添花
- 多并发时的工具限速（semaphore）
- `reflector` 覆盖度低时只对缺失子问题补查（当前实现是无差别触发 fanout）
- Mem0 长期记忆（跨会话偏好）
- 报告 PDF 导出
- Streamlit UI 增加 reports/threads 历史浏览页（M5 已有 SSE 基础）
- 评测：并行跑 + 失败重试；judge 用 self-consistency（n=3 取均值）

---

## 九、历史会话关键对话摘要

| 时间 | 用户意图 | 产出 |
|---|---|---|
| 初次 | 写 ENGINEERING.md | 15 章工程设计文档 + MCP 双向设计 |
| 之后 | M1 骨架 | ~40 文件，空节点主图跑通 |
| 之后 | M2-M4 真实代码 | 7 agent + 4 工具 + API + Internal MCP handlers |
| 之后 | "ArXiv/BRAVE 都没填 key 为什么有数据" | 澄清：ArXiv 无需 key；Brave MCP 当时是 stub |
| 之后 | "换 DeepSeek/DashScope 替代 Brave" | 加 DashScopeSearchTool 兜底 |
| 之后 | "加回 Brave MCP 真实调用，给 key" | 实现 mcp_brave_tool + mcp_loader 真实注册 |
| 之后 | "升级 Python" | 切 3.11 env，去掉 sync 桥，换官方 mcp SDK |
| 之后 | "浏览器能访问 brave" | 确诊代理问题；用户选择不接代理 |
| 现在 | 生成进度日志 | 本文件 |

---

## 十、最后一次真实运行证据

```
thread_id=f538bf53
Planner: 5 子问题（含 web/academic/code/kb 多源）
ArXiv: 每路命中 5 条
Tavily: 每路命中 5 条
Writer: qwen-max 输出 3800+ 字 Markdown，[^1]-[^13] 引用
归档: data/reports/20260418-180845_*.md
```

14 单测全绿 · 0 报错 · 0 警告（除 1 条 langchain 的 type hint 建议）。
