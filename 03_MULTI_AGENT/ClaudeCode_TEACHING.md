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

## 三、完整流程流转（代码对照版）

### 首轮研究（Turn 1）

**Step 1 — 用户启动研究**
```python
# app/api.py
@app.post("/research")
async def start_research(req: StartReq):
    tid = uuid.uuid4().hex[:12]                      # 新会话的 thread_id
    cfg = {"configurable": {"thread_id": tid}}       # 喂给 checkpointer
    payload = {"research_query": req.research_query, "audience": req.audience,
               "messages": [], "evidence": []}
    result = await bootstrap.app_state.graph.ainvoke(payload, config=cfg)
    interrupt_val = _extract_interrupt(result)        # 命中 interrupt 就返回
```
*关键点*：`thread_id` 是跨会话复用状态的唯一钥匙；`payload` 是 State 的初始 patch。

**Step 2 — Planner LLM 拆子问题 + `interrupt()` 暂停**
```python
# agents/planner.py
async def planner_node(state):
    llm = get_llm("max", temperature=0.3)
    structured = llm.with_structured_output(ResearchPlan, method="function_calling")
    plan = await structured.ainvoke([SystemMessage(...), HumanMessage(...)])
    decision = interrupt({"phase": "plan_review", "plan": plan.model_dump()})  # ← 暂停
    confirmed = _coerce_plan(decision, fallback=plan)
    return {"plan": confirmed.sub_questions, "plan_confirmed": True, ...}
```
*关键点*：`interrupt()` 抛出特殊异常，LangGraph runtime 捕获后把图挂起，payload 写进 `__interrupt__`。

**Step 3 — API 把 interrupt 回传给前端**
```python
# app/api.py
def _extract_interrupt(result):
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    item = intr[0] if isinstance(intr, list) else intr          # LangGraph 0.6+ 是 list
    return getattr(item, "value", None)
```

**Step 4 — 用户编辑后 resume**
```python
# app/api.py
@app.post("/research/{thread_id}/resume")
async def resume_research(thread_id, req):
    cfg = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(Command(resume={"plan": req.plan.model_dump()}), cfg)
```
*关键点*：`Command(resume=...)` 会变成 `interrupt()` 的返回值；checkpointer 按 `thread_id` 自动加载暂停态。

**Step 5 — Supervisor fan-out 派发 4 路 researcher**
```python
# graph/router.py
def supervisor_route(state):
    if not state.get("plan_confirmed"):  return "planner"
    if state.get("revision_count", 0) >= 3:  return "writer"    # 硬兜底
    if plan and not state.get("evidence") or state.get("next_action") == "need_more_research":
        sends = []
        for sq in state["plan"]:
            if sq.status == "done": continue                     # 已答过的 sub-q 跳过
            for src in sq.recommended_sources or ["web"]:
                node = _SOURCE_TO_NODE[src]                      # web→web_researcher 等
                sends.append(Send(node, {"sub_question": sq,
                                          "research_query": state["research_query"]}))
        return sends or "writer"
```
*关键点*：返回 `list[Send]` = 并行调度；payload 只喂给被派发的子节点，不污染主 state。

**Step 6 — 4 个 Researcher 并行取证 + merge_evidence 聚合**
```python
# agents/researcher_web.py
@with_tags("web_researcher")
@safe_node
async def web_researcher_node(payload):
    sq_id, question = extract_sq_and_query(payload)
    evidence = await run_research_chain(source_type="web", query=question,
                                         sub_question_id=sq_id,
                                         registry=app_state.registry, top_k=5)
    return {"evidence": evidence, ...}
```
*关键点*：4 个节点返回的 `evidence` 在 fan-in 时自动走 `merge_evidence` reducer —— 按 URL 去重 + 按 score 倒序。

**Step 7 — Reflector 打分 + 决定补查/收敛**
```python
# agents/reflector.py
async def reflector_node(state):
    rc = state.get("revision_count", 0) + 1
    if rc >= MAX_REVISION:                                       # 硬兜底：不调 LLM
        return {"next_action": "force_complete", "revision_count": rc, ...}
    # 按 sub_question_id 分组 + 截断前 5 条，控制 LLM 输入 token
    by_sq = {}
    for ev in state["evidence"]:
        by_sq.setdefault(ev.sub_question_id, []).append(...)
    result = await llm.with_structured_output(ReflectionResult, method="function_calling") \
                    .ainvoke([SystemMessage(...), HumanMessage(...)])
    return {"next_action": result.next_action, "coverage_by_subq": ...,
            "revision_count": rc, ...}
```

**Step 8 — reflector_route 决定走向**
```python
# graph/router.py
def reflector_route(state):
    if state.get("revision_count", 0) >= 3:                      # 双重兜底
        return "writer"
    return "supervisor" if state["next_action"] == "need_more_research" else "writer"
```

**Step 9 — Writer 生成 Markdown + 引用 + 落盘**
```python
# agents/writer.py
numbered, citations = [], []
for i, ev in enumerate(evidence, 1):                             # 后端编号 ↔ url 一一对应
    numbered.append(f"[{i}] ({ev.source_type}) {ev.source_url}\n    {ev.snippet[:400]}")
    citations.append(Citation(idx=i, source_url=ev.source_url, title=ev.snippet[:80]))
resp = await llm.ainvoke([SystemMessage(WRITER_SYSTEM), HumanMessage(...)])  # 要求用 [^N] 引用
report_md = resp.content.strip()
if not _has_citation_section(report_md):                         # LLM 漏写就自动补
    report_md += "\n\n## 引用\n" + "\n".join(f"[^{c.idx}]: {c.source_url}" for c in citations)
path = report_store.save(query, thread_id, report_md)            # 归档到 data/reports/
```
*关键点*：**编号由后端生成**，LLM 只负责在正文用 `[^1] [^2]`；即使 LLM 漏写引用章节，后端也会兜底补上。

### 追问（Turn 2，同 thread_id）

```python
# app/api.py
@app.post("/research/{thread_id}/turn")
async def turn_research(thread_id, req):
    patch = reset_per_turn({}, req.research_query)   # 重置 revision_count/next_action/...
    patch["plan_confirmed"] = False                  # 强制回到 planner 重拆
    result = await graph.ainvoke(patch, config=cfg)  # checkpointer 自动合并历史 state
```

```python
# app/turn_init.py
PER_TURN_RESET_FIELDS = ("revision_count", "iteration", "next_node",
                          "coverage_by_subq", "missing_aspects", "next_action")
def reset_per_turn(state, new_query):
    patch = {"research_query": new_query}
    for k in PER_TURN_RESET_FIELDS:
        patch[k] = 0 if k in {"revision_count", "iteration"} else None
    patch["coverage_by_subq"] = {}; patch["missing_aspects"] = []
    return patch
```

*关键点*：`evidence / plan / messages / final_report` 都**保留**。新一轮 Planner 拆出来的 sub_question 如果 URL 在历史 evidence 里命中，`merge_evidence` 会自动复用；没命中的才触发新检索。这是"多轮追问证据复用"的核心机制。

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
class ResearchState(TypedDict, total=False):
    research_query: str
    audience: str

    plan: list[SubQuestion]
    plan_confirmed: bool

    evidence: Annotated[list[Evidence], merge_evidence]   # ← 自定义 reducer
    revision_count: int

    coverage_by_subq: dict[str, int]
    missing_aspects: list[str]
    next_action: str                     # sufficient / need_more_research / force_complete
    additional_queries: list[str]

    final_report: str
    citations: list[Citation]
    report_path: str

    messages: Annotated[list[BaseMessage], add_messages]   # 官方 reducer，追加不覆盖
    iteration: int
    current_node: str
```

```python
def merge_evidence(old, new) -> list[Evidence]:
    """按 source_url 去重 + 按 relevance_score 倒序。兼容 Pydantic / dict 互转。"""
    pool = list(old or []) + list(new or [])
    by_url: dict[str, dict] = {}
    for raw in pool:
        d = _to_dict(raw)                       # Pydantic.model_dump() 或 dict()
        url = d.get("source_url") or ""
        if not url: continue
        cur = by_url.get(url)
        if cur is None or d["relevance_score"] > cur["relevance_score"]:
            by_url[url] = d                     # 同 URL 留 score 最高
    merged = [Evidence(**d) for d in by_url.values()]
    merged.sort(key=lambda e: -float(e.relevance_score or 0.0))
    return merged
```

**关键点**：`Annotated[list, merge_evidence]` 让 LangGraph 在每次 fan-in 时调用 reducer 合并多个并行节点返回的 evidence。`total=False` 表示所有字段可选（节点只返增量 patch）。

**为什么自定义 reducer 而不是 `operator.add`？**
- 4 个 researcher 可能爬到同 URL（如 Tavily + DashScope 同时命中同一篇博文），`operator.add` 只会简单拼接留重复。
- `merge_evidence` 三步语义：**按 URL 聚合 → 同 URL 保留 score 最高 → 按 score 倒序输出**，给 Writer 的是一张干净的高质量证据表。
- 必须兼容 Pydantic 对象和 dict —— LangGraph checkpointer 序列化为 JSON 再反序列化时会变成 dict，reducer 两种形态都要能处理。

### 2. Graph 装配 —— `graph/workflow.py`

```python
wf = StateGraph(ResearchState)
wf.add_node("planner", planner_node)
wf.add_node("supervisor", supervisor_node)
wf.add_node("web_researcher", web_researcher_node)
wf.add_node("academic_researcher", academic_researcher_node)
wf.add_node("code_researcher", code_researcher_node)
wf.add_node("kb_researcher", kb_researcher_node)
wf.add_node("reflector", reflector_node)
wf.add_node("writer", writer_node)

wf.add_edge(START, "planner")
wf.add_edge("planner", "supervisor")

wf.add_conditional_edges("supervisor", supervisor_route,
    {"planner":"planner", "writer":"writer",
     "web_researcher":"web_researcher", "academic_researcher":"academic_researcher",
     "code_researcher":"code_researcher", "kb_researcher":"kb_researcher"})

# 4→1 fan-in：所有 researcher 都收敛到 reflector
for r in ("web_researcher", "academic_researcher", "code_researcher", "kb_researcher"):
    wf.add_edge(r, "reflector")

wf.add_conditional_edges("reflector", reflector_route,
    {"supervisor":"supervisor", "writer":"writer"})
wf.add_edge("writer", END)
return wf.compile(checkpointer=checkpointer)
```

**关键点**：
- `add_conditional_edges` 的第 3 个参数是 **target 白名单 dict** —— 必须显式列出 router 可能返回的所有名字，否则 LangGraph 不放行。
- `supervisor → researchers` 的"显式列出 4 个 target"是为了让 Send 派发能命中；fan-in 端用静态 `add_edge` 即可，无需 router。
- `reflector` 虽然只有两个出口，也必须用 `add_conditional_edges`（不是 `add_edge`），否则无法根据 state 动态选。

### 3. 路由 —— `graph/router.py`（真实三分支）

```python
_SOURCE_TO_NODE = {"web":"web_researcher", "academic":"academic_researcher",
                   "code":"code_researcher", "kb":"kb_researcher"}

def supervisor_route(state):
    # 分支 1：计划未确认 → 回 planner（HITL 恢复后重入）
    if not state.get("plan_confirmed"):
        return "planner"
    # 分支 2：revision_count 超限 → 直接 writer（硬兜底防死循环）
    if state.get("revision_count", 0) >= 3:
        return "writer"
    # 分支 3：首轮 or Reflector 要求补查 → fan-out
    plan = state.get("plan") or []
    if plan and not state.get("evidence") or state.get("next_action") == "need_more_research":
        sends = []
        for sq in plan:
            if getattr(sq, "status", "pending") == "done":  continue
            for src in (sq.recommended_sources or ["web"]):
                node = _SOURCE_TO_NODE.get(src)
                if not node:  continue
                sends.append(Send(node, {"sub_question": sq,
                                          "research_query": state["research_query"]}))
        return sends or "writer"
    return "writer"

def reflector_route(state):
    if state.get("revision_count", 0) >= 3:  return "writer"      # 双重兜底
    action = state.get("next_action", "sufficient")
    return "supervisor" if action == "need_more_research" else "writer"
```

**关键点**：
- `Send(target, payload)` 是 LangGraph 的 fan-out 原语；**payload 只喂被派发的子节点**，不污染主 state。
- `sq.recommended_sources` 是 Planner 在拆子问题时就决定的（如 `["web","academic"]`），路由按此派发到对应 researcher；一个 sub-q 可以同时派给多个 source。
- **双重兜底**：`supervisor_route` 和 `reflector_route` 都检查 `rc>=3`，任何一端触发都会收敛到 writer。

### 4. Planner + HITL —— `agents/planner.py`

```python
async def planner_node(state):
    query = state.get("research_query", "")
    audience = state.get("audience", "intermediate")
    llm = get_llm("max", temperature=0.3)
    structured = llm.with_structured_output(ResearchPlan, method="function_calling")
    plan = await structured.ainvoke(
        [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=planner_user(query, audience))]
    )
    decision = interrupt({"phase": "plan_review", "plan": plan.model_dump()})
    confirmed = _coerce_plan(decision, fallback=plan)
    return {"plan": confirmed.sub_questions, "plan_confirmed": True,
            "iteration": 0, "revision_count": 0, "current_node": "planner", "messages": [...]}

def _coerce_plan(decision, *, fallback):
    """resume 的数据形态兼容：None / ResearchPlan / {"plan": {...}} / dict。"""
    if decision is None:  return fallback                    # 用户直接 resume 不改
    if isinstance(decision, ResearchPlan):  return decision
    if isinstance(decision, dict):
        payload = decision.get("plan", decision)
        if isinstance(payload, dict) and "sub_questions" in payload:
            try: return ResearchPlan.model_validate(payload)
            except Exception as e:  logger.warning("resume 解析失败: %s", e)
    return fallback
```

**关键点**：
- 节点执行到 `interrupt()` 会**挂起**，首次运行到此终止；resume 时 LangGraph **重新跑整个节点**，但 `interrupt()` 直接返回 `Command(resume=...)` 里的值。所以 `_coerce_plan` 必须能认多种形态。
- `plan_confirmed=True` 是 resume 后才设的 —— 不需要在节点开头检查"已确认就跳过"，因为 LangGraph 保证一个 interrupt 点只会对应一次 resume。

**踩坑**：`method="function_calling"` 必须显式传。DashScope compat 端点的默认 `json_object` 模式要求 prompt 含 "json" 字样，否则报 `'messages' must contain 'json'`。

### 5. safe_node 装饰器 —— `agents/_safe.py`（真实版本）

```python
def safe_node(fn):
    @functools.wraps(fn)
    async def wrapper(state, *args, **kwargs):
        try:
            return await fn(state, *args, **kwargs)
        except Exception as e:
            logger.warning("node %s failed: %s", fn.__name__, e, exc_info=True)
            return {"evidence": [], "messages": [AIMessage(content=f"[skip] {fn.__name__}: {e}")]}
    return wrapper
```

**作用**：单个 researcher 挂掉返回**空 evidence + 警告消息**，不中断主图；reflector 看到某 sub_q 覆盖度为 0 会自动触发补查。

**注意**：默认 patch 是**写死**的（空 evidence），不接受参数。这是刻意简化：所有 researcher 的失败态语义一致，没必要让每个节点自定义。

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
    # 1. 工具注册（顺序即 web 降级链：Tavily 主 → Brave MCP → DashScope 兜底）
    registry = ToolRegistry()
    registry.register(TavilyTool(api_key=settings.tavily_api_key))
    for tool in await load_external_mcp(settings.mcp_config_path):
        registry.register(tool)                              # Brave MCP
    registry.register(DashScopeSearchTool())                 # 国内兜底
    registry.register(ArxivTool())                           # academic
    registry.register(GitHubTool(token=settings.github_token))  # code
    registry.register(KBRetriever())                         # kb（01_RAG 复用）
    app_state.registry = registry

    # 2. AsyncSqliteSaver 是 async context manager，必须走 enter_async_context
    app_state._exit_stack = AsyncExitStack()
    cm = AsyncSqliteSaver.from_conn_string(str(db_path))
    checkpointer = await app_state._exit_stack.enter_async_context(cm)

    # 3. 编译图（checkpointer 传进去，所有节点完成后自动 snapshot）
    app_state.graph = build_graph(checkpointer=checkpointer)

async def shutdown():
    await app_state.registry.close_all()                     # 关 HTTP client / MCP 子进程
    await app_state._exit_stack.aclose()                     # 关 checkpointer 的 sqlite 连接
```

**关键点**：
- **每个工具的 register 都用 `try/except` 包裹**（省略未写），单个工具初始化失败不阻止启动 —— 降级链里少一环也能跑。
- `AsyncExitStack.aclose()` 会**逆序**调用所有已注册的 `__aexit__`，把底层 aiosqlite 连接、文件句柄全部关干净。
- 注册顺序 = 降级顺序，这是 `ToolRegistry._tools[source].append` 的副作用。**改变顺序就改变降级行为**，这是个隐式契约，新人接手容易踩。

### 9. FastAPI 三端点 —— `app/api.py`

| 端点 | 作用 | 关键代码 |
|---|---|---|
| `POST /research` | 首次启动 | `ainvoke(payload, cfg)` → 检查 `__interrupt__` |
| `POST /research/{tid}/resume` | 恢复 | `ainvoke(Command(resume={"plan":...}), cfg)` |
| `POST /research/{tid}/turn` | 追问 | `reset_per_turn + plan_confirmed=False` |

**interrupt 真实提取**（LangGraph 0.6+ 返回的是 `list[Interrupt]`，要兼容两种形态）：
```python
def _extract_interrupt(result):
    intr = result.get("__interrupt__") if isinstance(result, dict) else None
    if not intr:  return None
    item = intr[0] if isinstance(intr, list) else intr
    return getattr(item, "value", None) or (item.get("value") if isinstance(item, dict) else None)
```

**lifespan 装配**（FastAPI 0.100+ 推荐方式，替代旧的 `on_event`）：
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await bootstrap.startup()
    yield                              # ← 应用运行期
    await bootstrap.shutdown()

app = FastAPI(title="InsightLoop", version="0.1.0", lifespan=lifespan)
```

### 10. Writer 引用机制 —— 后端编号 + `[^N]` 回填

**目的**：LLM 写长文本时引用编号容易错位（漏引、乱编）。这里用"后端发号，LLM 只负责回填"的契约。

```python
# agents/writer.py
numbered, citations = [], []
for i, ev in enumerate(evidence, 1):            # ← 编号由后端分配
    numbered.append(f"[{i}] ({ev.source_type}) {ev.source_url}\n    {ev.snippet[:400]}")
    citations.append(Citation(idx=i, source_url=ev.source_url, title=...))

# 把编号好的 evidence 喂给 LLM，prompt 里明确要求用 [^N] 引用
resp = await llm.ainvoke([SystemMessage(WRITER_SYSTEM),
                          HumanMessage(writer_user(query, audience, plan_summary, numbered_evidence))])
report_md = resp.content.strip()

# 兜底：LLM 如果漏写引用章节，后端从 citations 自动补
_CITATION_HEADER = re.compile(r"^##\s*(引用|参考(文献)?|references?)", re.IGNORECASE | re.MULTILINE)
if not _CITATION_HEADER.search(report_md) and citations:
    report_md += "\n\n## 引用\n" + "\n".join(f"[^{c.idx}]: {c.source_url}" for c in citations)
```

**关键点**：**`citations` 数组从 evidence 直接构建**，编号 ↔ URL 严格一一对应；LLM 只负责在正文写 `[^1][^2]`，不参与编号分配。即使 LLM 把 `[^5]` 写错成 `[^15]`，后端的引用章节也是正确的（提供人工核对线索）。

### 11. MCP Brave —— stdio 子进程会话常驻

```python
# tools/mcp_brave_tool.py
class MCPBraveSearchTool:
    async def _ensure_session(self):
        if self._session is not None:  return self._session   # 复用
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(stdio_client(self._params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._stack, self._session = stack, session
            return session
        except Exception:
            await stack.aclose();  raise

    async def search(self, query, *, top_k=5):
        session = await self._ensure_session()
        result = await session.call_tool("brave_web_search", {"query": query, "count": top_k})
        return _parse_brave_text(_extract_text(result), top_k)

    async def close(self):
        if self._stack: await self._stack.aclose()
```

**为什么 session 常驻而不是每次重建**：
- `stdio_client` 冷启动要 `npx` 下载 + Node 启动，耗时 1-3 秒；高频搜索时这个代价分摊不值。
- MCP SDK 的 `ClientSession` 内部有请求锁，并发 `call_tool` 安全。
- `AsyncExitStack` 绑在实例上，`close()` 时一次性关闭 stdio 管道 + 杀子进程。

**国内可达性**：代码里会把 `HTTPS_PROXY` 传给子进程的 `env`，让 Node 的 fetch 走代理；不配代理时 Brave MCP 调用会超时，降级到下一个工具（DashScope 内置搜索）。

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

"三层兜底。第一层：state 里 `revision_count` 计数器；第二层：`MAX_REVISION=3` 硬上限，reflector 检测到立刻返 `force_complete` 不调 LLM；第三层：`supervisor_route` 和 `reflector_route` 都有 `if rc>=3: return 'writer'`，**两端都卡**。即使 evidence 不足也会出报告 —— 让 LLM 在报告里注明'信息有限'比无限循环强。生产系统里硬兜底永远比模型判断稳。"

### Q9：Writer 的引用编号怎么保证不乱？

"契约是'后端发号，LLM 只回填'。Writer 先遍历 `evidence` 数组，按顺序给每条分配编号 `i=1,2,...`，同时构建 `Citation(idx=i, source_url=...)` 列表。喂给 LLM 的 prompt 里 evidence 已带编号 `[1] [2] ...`，系统提示要求正文用 `[^N]` 引用。即使 LLM 漏写引用章节，后端用正则 `^##\s*(引用|参考|references?)` 检测，没找到就自动从 `citations` 生成 `## 引用\n[^1]: url\n[^2]: url\n...`。编号 ↔ URL 的一致性由后端保证，LLM 只负责语义层引用。"

### Q10：MCP Brave 的 stdio 子进程怎么管？

"用官方 mcp SDK 的 `AsyncExitStack` + `ClientSession`。第一次 `search()` 时 `_ensure_session()` 启动 `npx @modelcontextprotocol/server-brave-search` 子进程，建立 stdio 管道，创建 ClientSession 并 `initialize()`，整个栈绑在实例 `self._stack` 上。后续 `search()` 直接复用 session —— stdio 冷启动要 1-3 秒，每次重建太贵。并发安全由 SDK 内部的请求锁保证。`close()` 时一次 `aclose()` 关管道 + 杀子进程。`registry.close_all()` 会调到它，和 FastAPI lifespan 的 shutdown 钩上。"

### Q11：同源 URL 在多个 researcher 里命中怎么办？

"`merge_evidence` reducer 处理。fan-in 时 LangGraph 把 4 路 researcher 返回的 evidence 列表合并，reducer 按 `source_url` 分组，同 URL 多次命中**保留 `relevance_score` 最高**的一条，最终按 score 倒序输出。这是 `Annotated[list[Evidence], merge_evidence]` 的 reducer 契约。用 `operator.add` 也能合并，但会保留重复 —— Writer 就会看到同篇博文两次引用，引用编号会重复。所以必须自定义。"

---

---

## 八、M5 实时事件流 + Streamlit 聊天 UI

> 把"接口调用 → 干等几十秒"升级为"提交 → 实时看到 8 节点逐个跑完 → writer 流式渲染 markdown"。

### 8.1 设计要点：interrupt 不在事件流里

**踩坑记录**：第一版用 `astream_events(version="v2")` 监听 `__interrupt__` 字段，怎么都拿不到。`astream_events` 在 interrupt 触发时：
- 被截断的节点（planner）**不发** `on_chain_end`
- 顶层 `LangGraph` 的 `on_chain_end` output 里**也没有** `__interrupt__` 字段
- interrupt 信息**只能**通过 `graph.aget_state(cfg).tasks[*].interrupts` 拿

```python
# app/sse.py
async def stream_events(events, *, thread_id, graph, cfg):
    yield {"event": "thread", "data": {"thread_id": thread_id}}
    async for ev in events:
        mapped = map_event(ev)
        if mapped: yield mapped
    pending = await _detect_pending_interrupt(graph, cfg)   # 关键：流结束后再问 state
    if pending:
        yield {"event": "interrupt", "data": pending}
        return
    yield {"event": "done", "data": {...}}
```

### 8.2 Writer-only token 流

planner / reflector 也调 LLM，但它们的中间 token 不应该轰炸前端：

```python
# app/sse.py:map_event
if etype == "on_chat_model_stream":
    if metadata.get("langgraph_node") != "writer":
        return None
    return {"event": "token", "data": {"text": chunk.content}}
```

### 8.3 三条 SSE 端点

```
GET  /research/stream?query=...&audience=...      启动 + 流推
GET  /research/{tid}/resume_stream?plan=<json>    用户编辑 plan 后恢复
GET  /research/{tid}/turn_stream?query=...        同会话追问
```

旧的 `/research`、`/resume`、`/turn`（POST）全部保留 —— 测试和 Internal MCP 在用。

### 8.4 Streamlit 单页 chat UI

`app/streamlit_ui.py`：8:4 双栏。
- 左：报告区（writer token 累积渲染，每 12 个 token 刷一次 placeholder，避免 rerun 风暴）
- 右：实时进度（按 graph 顺序，**只显示 plan 中用到的 researcher**：从 `sub_questions[*].recommended_sources` 推出来）
- interrupt 弹层：`st.expander + st.data_editor`；点确认时立即 `plan_panel.empty()` 清掉按钮

**使用**：
```bash
PYTHONPATH=. uvicorn app.api:app --port 8080      # 后端
streamlit run app/streamlit_ui.py                  # UI
```

---

## 九、M6 LangSmith 自动追踪 + LLM-as-judge 评测

### 9.1 LangSmith：零侵入开启

LangChain 全局 callback 检测到 `LANGCHAIN_TRACING_V2=true` + key 时自动上报，**不需要改任何节点代码**。

```python
# app/bootstrap.py
def _setup_langsmith():
    if not settings.langchain_tracing_v2 or not settings.langchain_api_key:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
```

**业务标签**：在 `RunnableConfig.metadata` 里塞 thread_id / case_id 等，云端可按字段筛 trace：

```python
# app/api.py
def _config(tid, *, query=None, audience=None):
    return {
        "configurable": {"thread_id": tid},
        "metadata": {"thread_id": tid, "research_query": query, "audience": audience, "app": "insightloop"},
    }
```

**为什么不在节点上手动加 tag**：LangGraph 自动把节点名写进 run name，足够 filter；`config/tracing.py:with_tags` 装饰器先保留作钩子，需要更细维度时再启用。

### 9.2 LLM-as-judge：三维度结构化打分

`evals/judge.py` 用 DeepSeek pro `with_structured_output(JudgeScore, method="json_mode")` 给报告打分：

```python
class JudgeScore(BaseModel):
    coverage: int      # 0-100  plan 中的 sub_questions 是否被报告答到
    accuracy: int      # 0-100  关键论断是否能在 evidence 里找到支撑
    citation: int      # 0-100  [^N] 引用是否对应 evidence
    overall: int       # 加权 0.4·cov + 0.3·acc + 0.3·cit
    rationale: str     # 中文一段话，每维度给 1 句解释 + 失分点
```

prompt 同时塞 `query / plan / evidence_brief / report_md`，让 judge 看到"问题—调研—成稿"三方，而不是只评最终文本。报告超 6000 字自动截断防止吃满上下文。

### 9.3 评测脚本：interrupt 自动接受

评测场景下不需要 HITL，自动 accept LLM 给的 plan：

```python
# evals/run.py
r1 = await g.ainvoke({"research_query": case["query"], ...}, config=cfg)
proposed = r1["__interrupt__"][0].value
r2 = await g.ainvoke(Command(resume={"plan": proposed["plan"]}), config=cfg)
score = await judge_one(JudgeInput(query=..., plan=..., evidence_brief=..., report_md=r2["final_report"]))
```

特性：
- `--limit N` 烟测；`--dataset path` 自定义集
- 失败案例不中断整轮，`error` 字段记 `{ExceptionType}: {msg}`
- 每条 case 都给 `tags=["eval"]` + `metadata.eval_run_id` —— LangSmith 上一眼就能把评测 trace 与 demo trace 分开

产出：`evals/results/{run_id}/results.jsonl` + `REPORT.md`

### 9.4 Markdown 报告 + Streamlit 看板：双产物

| 产物 | 文件 | 用途 |
|---|---|---|
| Markdown | `REPORT.md` | 静态总结（PR/汇报直接贴），含表 + 维度均值 + 失分案例 |
| 看板 | `app/evals_ui.py` | 交互看板：单 run 详情 + **两 run 对比柱状图** + 单题 rationale + 报告全文 |

看板的"对比模式"是面试演示亮点 —— 同一数据集跑两次（比如改了 reflector 阈值前后），能直观看维度均值变化。

### 9.5 完整使用方式

**前置**：在 `03_MULTI_AGENT/.env` 设 LangSmith key：
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=insightloop-multi-agent
```

**1. 跑评测**（5 题约 5-10 分钟，调真 LLM 和搜索）：
```bash
conda activate langgraph-cc-multiagent
cd 03_MULTI_AGENT

# 烟测 1 题（~1 分钟）
PYTHONPATH=. python -m evals.run --limit 1

# 全量 5 题
PYTHONPATH=. python -m evals.run

# 自定义数据集
PYTHONPATH=. python -m evals.run --dataset evals/my_set.jsonl
```

产出：
```
evals/results/20260426-153000/
  ├── results.jsonl   # 每行一个 case 的输入/报告/打分/耗时
  └── REPORT.md       # 可读评测报告
```

**2. 看 Markdown 报告**：直接打开 `evals/results/{run_id}/REPORT.md`

**3. 启 Streamlit 看板**：
```bash
streamlit run app/evals_ui.py
```
- 侧栏切换"单 run 详情"/"两 run 对比"
- 单 run：4 项指标卡 + 维度均值柱状图 + 用例 dataframe + 单条详情（含报告全文 expander）
- 对比：选两个 run，并排维度均值柱状图 + 逐用例 overall delta

**4. 看 LangSmith trace**：登 https://smith.langchain.com → 选 project `insightloop-multi-agent` → filter `metadata.eval_run_id = "20260426-153000"`

### 9.6 数据集扩展指南

`evals/dataset.jsonl` 一行一题：
```json
{"id":"tech_01","category":"技术","query":"...","audience":"intermediate"}
```

DEV_PROGRESS 原计划 20 题 = 5 × {技术、产业、对比、追问}。当前是 5 题烟测，扩到 20 直接往 jsonl 加行即可，无需改任何代码。

### 9.7 面试可讲点

- **不写一行节点代码就接入了 trace**：靠 LangChain 全局 callback + RunnableConfig.metadata 的两层设计；展示了对 LangChain 抽象的理解
- **judge 看到三方信息（query / plan / evidence / report）而不是只看 report**：避免"报告写得漂亮但实际偏题"的盲区
- **失败不中断整轮 + tags=["eval"]**：评测产物可对比、可复现、与 demo trace 隔离 —— 工程化思维
- **本地 + 云端双轨**：Markdown/看板满足无网演示，LangSmith 满足深度归因 —— 面向不同观众

---

## 十、快速参考卡

**启动 3 行**：
```bash
conda activate langgraph-cc-multiagent
cd 03_MULTI_AGENT
PYTHONPATH=. python -m scripts.run_local "研究问题"
```

**关键文件导航**：
- State 契约 / reducer：`graph/state.py` `merge_evidence` L32-49
- 图装配 / fan-in：`graph/workflow.py` `build_graph`
- 三分支路由：`graph/router.py` `supervisor_route` L22-54
- HITL + resume 兼容：`agents/planner.py` `_coerce_plan` L44-58
- 安全装饰器：`agents/_safe.py` `safe_node`（默认返回空 evidence）
- 降级链：`agents/_researcher_base.py` `run_research_chain` L19-52
- 双重硬兜底：`agents/reflector.py` `MAX_REVISION=3` + `reflector_route` 的 rc 检查
- Writer 引用回填：`agents/writer.py` `_has_citation_section` + 兜底补引用
- 生命周期：`app/bootstrap.py` `AsyncExitStack.enter_async_context`
- FastAPI：`app/api.py` `_extract_interrupt` + `Command(resume=...)`
- MCP stdio 会话：`tools/mcp_brave_tool.py` `_ensure_session`
- KB 复用手术：`tools/kb_retriever.py` `_load_01rag_get_hybrid_retriever`
- **SSE 事件映射**：`app/sse.py` `map_event` + `_detect_pending_interrupt`
- **Streamlit chat UI**：`app/streamlit_ui.py` `_active_nodes_from_plan`
- **LangSmith 接入**：`app/bootstrap.py:_setup_langsmith` + `app/api.py:_config` 的 metadata
- **judge**：`evals/judge.py` `JudgeScore` + `judge_one`
- **评测看板**：`app/evals_ui.py`

**记住三个数字**：
- **7** 个 Agent（Planner/Supervisor/4 Researchers/Reflector/Writer）
- **3** 轮 Reflexion 硬上限（supervisor_route + reflector_route 双重卡点）
- **5** 个 Internal MCP 工具对外暴露（kb_search / list_reports / read_report / list_evidence / trigger_research）
