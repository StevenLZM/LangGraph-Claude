# InsightLoop 多智能体系统 —— 教学文档

> 对照代码讲流程、讲设计、讲踩坑、讲面试答法。读完即可独立讲清这个项目。

---

## 一、项目全景

**定位**：生产级 AI 行业深度研究多 Agent 系统，面试作品集。

**一句话介绍**：用户输入研究问题 → Planner LLM 拆解为 5 个子问题 → 人工确认计划（HITL）→ 4 个 Researcher 并行检索（web/academic/code/kb）→ Reflector 评估证据覆盖度决定是否补查 → Writer 输出带引用的 Markdown 报告。

**三大亮点**（面试可讲）：
1. **Supervisor 中心调度**的 7-Agent 架构（LangGraph StateGraph + Send API fan-out）
2. **Human-in-the-Loop**：`interrupt()` + `Command(resume=...)` 支持用户编辑研究计划
3. **MCP 双向集成**：作为 client 调外部 Brave Search，作为 server 暴露 5 个工具给 Claude Desktop

---

## 二、架构分层

```
┌─────────────────────────────────────────────────┐
│  API 层     app/api.py  FastAPI async lifespan   │
├─────────────────────────────────────────────────┤
│  图编排     graph/        StateGraph + Router    │
├─────────────────────────────────────────────────┤
│  Agent 层   agents/       7 节点（LLM 调用）      │
├─────────────────────────────────────────────────┤
│  工具层     tools/        Registry + 降级链      │
├─────────────────────────────────────────────────┤
│  基础设施   config/ app/bootstrap.py 生命周期     │
└─────────────────────────────────────────────────┘
```

---

## 三、完整流程流转

### 首轮研究（Turn 1）

```
 用户 POST /research
   ↓
 graph.ainvoke(payload, cfg={thread_id})
   ↓
 [planner]  LLM 拆子问题 → interrupt() 暂停
   ↓                        (返回 Interrupt 给 API → HTTP 响应)
 用户编辑 plan
   ↓
 POST /research/{tid}/resume  →  Command(resume={plan:...})
   ↓
 [planner 从 interrupt 恢复] plan_confirmed=True
   ↓
 [supervisor] iteration++, current_node="supervisor"
   ↓
 supervisor_route(state) → [Send('researcher_web', sq1),
                            Send('researcher_academic', sq2), ...]
   ↓ 并行
 [researcher_web] [researcher_academic] [researcher_code] [researcher_kb]
   每个节点通过 registry.get_chain(source_type) 拿降级链，顺序尝试，首个非空即返回
   返回 evidence 列表 → merge_evidence reducer 按 URL 去重 + 按 score 倒序
   ↓ 汇聚
 [reflector]  LLM 打覆盖度分
    if score >= 0.75 → next_action="sufficient"
    elif rc >= 3     → "force_complete"  （硬兜底不调 LLM）
    else             → "need_more_research" → 回 supervisor
   ↓ (sufficient / force_complete)
 [writer]  qwen-max 生成 Markdown + 补充引用章节 + 归档 data/reports/
   ↓
 END → final_report / report_path 回到 API 响应
```

### 追问（Turn 2，同 thread_id）

```
 POST /research/{tid}/turn
   ↓
 reset_per_turn(patch, new_query)   # 保留 evidence/plan/messages；重置 revision_count/next_action
 patch["plan_confirmed"] = False    # 强制回到 planner
   ↓
 graph.ainvoke(patch, cfg={thread_id})  # checkpointer 自动加载历史 state
   ↓
 走同样的 planner → supervisor → ... → writer 流程
 但 evidence 中已有历史证据 → Writer 可复用
```

---

## 四、核心技术栈

| 层 | 选型 | 原因 |
|---|---|---|
| Agent 框架 | LangGraph 1.1 | StateGraph/Send/interrupt 原生支持多 Agent |
| LLM | qwen-max / qwen-plus (DashScope) | 国内可达，中文强 |
| LLM SDK | `langchain-openai.ChatOpenAI` | DashScope 兼容 OpenAI 协议 |
| 结构化输出 | `with_structured_output(..., method="function_calling")` | DashScope compat 端点不支持 json_object |
| HITL | `interrupt()` + `Command(resume=...)` | LangGraph 原生 |
| 持久化 | `AsyncSqliteSaver` + `aiosqlite` | 跨会话；async 契合 FastAPI |
| Web API | FastAPI + lifespan | async 全栈 |
| 外部工具 | Tavily / ArXiv / GitHub / Brave MCP / DashScope 内置搜索 | 多源 + 降级 |
| MCP SDK | 官方 `mcp` Python SDK | 标准协议 |
| KB 检索 | 复用 01_RAG (ParentChildHybridRetriever) | 向量 + BM25 + RRF |
| Python | 3.11 | MCP SDK + async contextvars |

---

## 五、代码对照讲解

### 1. State 契约 —— `graph/state.py`

```python
class ResearchState(TypedDict):
    research_query: str
    plan: list[SubQuestion]
    plan_confirmed: bool
    evidence: Annotated[list[Evidence], merge_evidence]   # ← 自定义 reducer
    revision_count: int
    next_action: Literal["sufficient","need_more_research","force_complete"]
    final_report: str
    ...
```

**关键点**：`Annotated[list, merge_evidence]` 让 LangGraph 在每次 fan-in 时调用 reducer 合并多个并行节点返回的 evidence。

**为什么自定义 reducer 而不是 `operator.add`？**
- 4 个 researcher 可能爬到同 URL，简单拼接留重复
- `merge_evidence(old, new)`：按 URL 聚合 → 同 URL 保留 score 最高 → 按 score 倒序输出
- 兼容 Pydantic 对象和 dict（LangGraph 序列化/反序列化会互转）

### 2. Graph 装配 —— `graph/workflow.py`

```python
g = StateGraph(ResearchState)
g.add_node("planner", planner_node)
g.add_node("supervisor", supervisor_node)
g.add_node("researcher_web", ...)
...
g.add_conditional_edges("supervisor", supervisor_route, [...])
g.add_conditional_edges("reflector", reflector_route, ["supervisor","writer"])
return g.compile(checkpointer=checkpointer)
```

**关键点**：`supervisor` 和 `reflector` 走条件边；并行 researchers 由 router 返回 `list[Send]` 触发。

### 3. 路由 —— `graph/router.py`

```python
def supervisor_route(state):
    if not state["plan_confirmed"]:
        return "planner"
    return [
        Send("researcher_web",      {"_sq": sq, "_query": sq.question})
        for sq in state["plan"] if sq.source_type=="web"
    ] + [...]   # academic/code/kb 同理

def reflector_route(state):
    if state["next_action"] in ("sufficient","force_complete"):
        return "writer"
    return "supervisor"
```

**关键点**：`Send(target, payload)` 是 LangGraph 的 fan-out 原语；payload 是该子节点的局部 state patch（不会污染主 state）。

### 4. Planner + HITL —— `agents/planner.py`

```python
async def planner_node(state, config=None):
    if state.get("plan_confirmed"):      # resume 后不再拆
        return {...}
    llm = get_llm("plus").with_structured_output(ResearchPlan, method="function_calling")
    plan = await llm.ainvoke([SystemMessage(...), HumanMessage(...)])
    edited = interrupt({"plan": plan.model_dump(), "hint": "可编辑后 resume"})
    final_plan = edited.get("plan") or plan.model_dump()
    return {"plan": final_plan["sub_questions"], "plan_confirmed": True, ...}
```

**踩坑**：`method="function_calling"` 必须显式传。DashScope compat 端点的默认 `json_object` 模式要求 prompt 含 "json" 字样，否则报错。

### 5. safe_node 装饰器 —— `agents/_safe.py`

```python
def safe_node(default_patch):
    def deco(fn):
        @wraps(fn)
        async def wrapped(state, config=None):
            try: return await fn(state, config)
            except Exception as e:
                logger.exception(...)
                return default_patch
        return wrapped
    return deco
```

**作用**：单个 researcher 挂掉不影响整图；返回空 evidence 即可，reflector 看到覆盖度低自动补查。

### 6. Researcher 降级链 —— `agents/_researcher_base.py`

```python
async def run_research_chain(registry, source_type, query, timeout=45):
    chain = registry.get_chain(source_type)   # e.g. web → [Tavily, BraveMCP, DashScope]
    for tool in chain:
        try:
            results = await asyncio.wait_for(tool.search(query), timeout=timeout)
            if results: return results         # 首个非空即返回
        except Exception: continue
    return []
```

**设计思想**：Protocol（duck typing）+ Registry 模式 —— 所有工具只要实现 `search()` 和 `source_type` 属性即可进链，无共同基类。

### 7. Reflector 硬兜底 —— `agents/reflector.py`

```python
MAX_REVISION = 3

async def reflector_node(state, config=None):
    rc = state.get("revision_count", 0)
    if rc >= MAX_REVISION:                    # 硬兜底：不调 LLM
        return {"next_action": "force_complete", ...}
    llm = get_llm("plus").with_structured_output(ReflectionResult, method="function_calling")
    result = await llm.ainvoke([...])
    return {"next_action": result.next_action, "revision_count": rc+1}
```

**为什么硬兜底？** 避免 LLM 判断不稳定导致死循环；3 轮后强制收敛到 writer（即使证据不够也要出报告，让用户看到"为什么不够"比卡死强）。

### 8. Bootstrap 生命周期 —— `app/bootstrap.py`

```python
async def startup():
    registry = ToolRegistry()
    registry.register(TavilyTool(...))                      # ← 注册顺序即降级顺序
    for t in await load_external_mcp(...): registry.register(t)   # Brave MCP
    registry.register(DashScopeSearchTool())                # 国内兜底
    registry.register(ArxivTool()); registry.register(GitHubTool(...)); registry.register(KBRetriever())

    app_state._exit_stack = AsyncExitStack()
    cm = AsyncSqliteSaver.from_conn_string(db_path)
    checkpointer = await app_state._exit_stack.enter_async_context(cm)
    app_state.graph = build_graph(checkpointer=checkpointer)
```

**关键点**：`AsyncExitStack` 统一管理 async context manager 的生命周期 —— `shutdown()` 一次 `aclose()` 关闭所有资源。

### 9. FastAPI 三端点 —— `app/api.py`

| 端点 | 作用 | 关键代码 |
|---|---|---|
| `POST /research` | 首次启动 | `ainvoke(payload, cfg)` → 检查 `__interrupt__` |
| `POST /research/{tid}/resume` | 恢复 | `ainvoke(Command(resume={"plan":...}), cfg)` |
| `POST /research/{tid}/turn` | 追问 | `reset_per_turn + plan_confirmed=False` |

**interrupt 提取**：`result["__interrupt__"][0].value` —— LangGraph 0.6+ 把 interrupt payload 以这个键放进 ainvoke 结果。

---

## 六、设计思想

### 1. 为什么选 Supervisor 而不是 Pipeline？

| 对比 | Pipeline | Supervisor |
|---|---|---|
| 灵活性 | 固定顺序 | 根据 state 动态路由 |
| 并行 | 难 | 天然支持 fan-out |
| 循环 | 不支持 | 条件边轻松循环 |
| 适用 | 流水线作业 | 研究/诊断类不定步骤任务 |

本项目要并行检索 + 循环补查，必须 Supervisor。

### 2. Reflexion 模式

Self-evaluation 闭环：`generate → critique → regenerate`。本项目中：
- generate：researcher 并行检索
- critique：reflector LLM 打覆盖度分
- regenerate：若不足回 supervisor 再次 fan-out
- **硬终止**：`revision_count >= 3`（工程化关键 —— 论文实现常忽略这点）

### 3. Protocol 而不是 ABC

```python
class SearchTool(Protocol):
    source_type: str
    async def search(self, query: str) -> list[ToolResult]: ...
```

好处：新增工具零继承成本；测试时 mock 对象也能通过 type check。

### 4. 01_RAG 复用的 sys.modules 手术

01_RAG 有自己的 `config/` 和 `mcp/` 子包，直接导入会污染本项目模块空间。解决方案：
1. 备份本项目 `config` 和 `mcp` 模块
2. 临时 `sys.path.insert(0, "01_RAG")` + `sys.modules.pop("config")`
3. 触发 `from rag.retriever import ...` —— 此时解析到 01_RAG 的 config
4. Python 把 `config` 绑定到 rag.retriever 的闭包内后，**finally 恢复 sys.path / sys.modules**
5. rag.retriever 后续调用仍能访问绑定的 01_RAG config（import 缓存），但全局环境已清洁

**不修改 01_RAG 任何文件** —— 这是给面试官展示"尊重 legacy 代码边界"的点。

---

## 七、面试高频问答

### Q1：介绍一下这个项目

"InsightLoop 是一个生产级多 Agent 研究系统，基于 LangGraph 构建。用户输入一个研究问题，系统通过 7 个 Agent 协作 —— Planner 拆解、Supervisor 调度、4 个并行 Researcher 从 web/学术/代码/知识库四路检索、Reflector 评估证据覆盖度决定是否补查、Writer 输出带引用的 Markdown 报告。亮点有三：一是 Supervisor 中心调度架构，用 LangGraph 的 Send API 实现 fan-out/fan-in 并行；二是支持 HITL，用户可以编辑 Planner 生成的研究计划；三是 MCP 双向集成 —— 既作为 client 调外部 Brave Search，也作为 server 把 5 个研究工具暴露给 Claude Desktop。"

### Q2：interrupt/resume 机制怎么实现的？

"LangGraph 原生支持。Planner 节点调 `interrupt(payload)` 会抛出一个特殊异常被 runtime 捕获，暂停图执行，把 payload 放进 `result['__interrupt__']` 返回给 API。API 检测到后响应 HTTP 给前端展示 plan。用户编辑后前端调 `/resume`，API 用 `Command(resume={'plan':...})` 再次 `ainvoke`，LangGraph 通过 `thread_id` 从 checkpointer 加载暂停态，把 `resume` 值作为 `interrupt()` 的返回值注入，节点从断点继续。持久化用 AsyncSqliteSaver。"

### Q3：怎么实现并行 Researcher？

"在 supervisor 的 conditional_edge 里返回 `list[Send]`：每个 Send 带 target node name 和局部 payload。LangGraph 检测到 list[Send] 会并行调度所有 target 节点。fan-in 时通过 State 字段上的 reducer 合并 —— evidence 字段用 `Annotated[list, merge_evidence]`，reducer 按 URL 去重 + score 倒序。"

### Q4：工具降级链怎么设计的？

"ToolRegistry 按 source_type 分组；注册顺序即降级顺序。`get_chain('web')` 返回 `[Tavily, BraveMCP, DashScope]`。researcher 里顺序 `try`，首个非空结果即返回，都空就该路本轮无证据。关键点是超时和异常隔离 —— 每个工具 `asyncio.wait_for(45s)` + `try/except`，单个工具挂不影响链。Brave MCP 国内不可达但代码完整，DashScope 内置搜索走原生端点国内可达兜底。"

### Q5：多轮对话 state 怎么管？

"thread_id 隔离会话；checkpointer 负责每轮后持久化。追问时调 `/turn`，`reset_per_turn` 有选择地重置：保留 evidence/plan/messages，重置 revision_count/next_action/current_node；同时把 plan_confirmed 设 False 让图重新走 planner。Planner 会看到历史 messages 决定是否复用旧 plan 或重新拆解。"

### Q6：为什么 DashScope 必须用 function_calling？

"DashScope 的 OpenAI 兼容端点对 structured output 有特殊行为 —— 默认 `json_object` 模式要求 prompt 里包含 'json' 字样，否则返回 `'messages' must contain 'json'` 错误。而 `function_calling` 模式用的是 tool_use 协议，不依赖 prompt 关键字。所以 `with_structured_output(Model, method='function_calling')` 必须显式指定。"

### Q7：MCP 双向集成怎么做的？

"作为 client：`tools/mcp_brave_tool.py` 用官方 mcp SDK 的 `ClientSession + stdio_client + AsyncExitStack` 启动 `npx -y @modelcontextprotocol/server-brave-search` 子进程，通过 stdio 传 JSON-RPC。`.mcp.json` 声明 server 配置，`mcp_loader` 解析后注册到 registry。作为 server：`tools/internal_mcp/server.py` 暴露 5 个工具（kb_search/list_reports/read_report/list_evidence/trigger_research），handlers 全真实实现，以 stdio 模式跑，Claude Desktop 配一行即可调用。"

### Q8：Reflector 死循环怎么防？

"三层兜底。第一层：state 里 `revision_count` 计数器；第二层：`MAX_REVISION=3` 硬上限，reflector 检测到立刻返 `force_complete` 不调 LLM；第三层：reflector_route 检测 `next_action in ('sufficient','force_complete')` 直接到 writer。即使 evidence 不足也会出报告 —— 让 LLM 在报告里注明'信息有限'比无限循环强。生产系统里硬兜底永远比模型判断稳。"

---

## 八、快速参考卡

**启动 3 行**：
```bash
conda activate langgraph-cc-multiagent
cd 03_MULTI_AGENT
PYTHONPATH=. python -m scripts.run_local "研究问题"
```

**关键文件导航**：
- State 契约：`graph/state.py` L10-30
- 图装配：`graph/workflow.py` build_graph
- 并行路由：`graph/router.py` supervisor_route
- HITL：`agents/planner.py` interrupt 调用
- 降级链：`agents/_researcher_base.py` run_research_chain
- 硬兜底：`agents/reflector.py` MAX_REVISION=3
- 生命周期：`app/bootstrap.py` startup/shutdown
- API：`app/api.py` /research /resume /turn

**记住三个数字**：
- **7** 个 Agent（Planner/Supervisor/4 Researchers/Reflector/Writer）
- **3** 轮 Reflexion 硬上限
- **5** 个 Internal MCP 工具对外暴露
