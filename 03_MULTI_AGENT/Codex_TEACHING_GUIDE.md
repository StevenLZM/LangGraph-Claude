# InsightLoop 项目教学文档

本文是给“想真正学会这个项目的人”准备的，不是给“只想看架构图的人”准备的。

你应该把它和下面这些文件一起看：

- `03_MULTI_AGENT/PROJECT_SCENARIO.md`
- `03_MULTI_AGENT/ENGINEERING.md`
- `03_MULTI_AGENT/DEV_PROGRESS.md`
- `03_MULTI_AGENT/graph/`
- `03_MULTI_AGENT/agents/`
- `03_MULTI_AGENT/tools/`
- `03_MULTI_AGENT/tests/`

先做一个勘误：

- 早先文档里写成了 `PROJECT SCENARIO.md`
- 实际仓库文件名是 `PROJECT_SCENARIO.md`

这份教学文档会严格以“当前真实代码”和“当前真实测试”为准，不把设计稿里尚未落地的内容讲成已交付能力。

---

## 1. 先用一句话讲清楚项目

`InsightLoop` 是一个基于 `LangGraph` 的多 Agent 深度研究系统，目标不是普通聊天，而是把复杂研究任务拆成一条可中断、可恢复、可并行、可补查、可归档的工程化流水线。

它解决的是这类问题：

- 用户的问题很大，不能一次性直接回答
- 需要先拆计划，再并行搜证据，再判断覆盖度，再写报告
- 中间要允许用户确认计划，避免方向跑偏
- 一轮报告写完之后，还要支持基于历史证据继续追问

所以你读这个项目时，不要把它理解成“7 个 agent 聊天协作”，而要理解成：

> 一个由显式状态机驱动的研究工作流，里面嵌了几个真正有价值的智能节点。

---

## 2. 学这个项目最正确的姿势

很多人读多 Agent 项目有两个常见错误：

- 从 `api.py` 开始逐文件平推，最后只记住一堆函数名
- 只看 `ENGINEERING.md`，不对照真实代码和测试，最后把未实现的能力也讲进面试

更高效的顺序应该是：

1. 先看 `PROJECT_SCENARIO.md`，理解它想解决什么业务问题
2. 再看 `ENGINEERING.md`，理解它原本想怎么设计
3. 然后按本文主线，从一次请求进入图开始，顺着 agent 流程读代码
4. 每读完一个阶段，立刻看对应测试，确认你的理解是“可验证的”
5. 最后再回头看技术选型、调优和面试表达

这份文档的组织方式也是按这个顺序来的。

---

## 3. 先对齐“设计稿”和“真实代码”

先把这件事讲透，不然后面容易读混。

### 3.1 当前代码真实已完成的能力

- `planner` 结构化拆计划
- `interrupt()` + `Command(resume=...)` 做 HITL 计划确认
- `supervisor_route()` 返回 `list[Send]` 做并行 fan-out
- 4 个 `researcher` 节点并行取证
- `merge_evidence()` 统一 fan-in 聚合
- `reflector` 做覆盖度判断和补查控制
- `writer` 生成报告并落盘
- `FastAPI` 提供 `/research`、`/resume`、`/turn`、`/state`
- `ToolRegistry` + fallback chain
- external MCP client + internal MCP server
- 多轮追问状态复用
- 离线闭环测试

### 3.2 设计里提到但当前还没完全落地的能力

- SSE 流式事件接口
- Streamlit UI
- Docker Compose 完整交付
- LangSmith 真正接线
- 更完整的自动评测体系

### 3.3 你必须知道的“文档和实现差异”

- `ENGINEERING.md` 里提到 `/research/{tid}/stream`，但 `app/api.py` 当前没有实现 SSE
- 调度真正发生在 `graph/router.py::supervisor_route()`，`agents/supervisor.py` 本身很薄
- `agents/writer.py` 顶部注释还写着旧说明，但真实实现已经是 `await llm.ainvoke(...)`
- `README.md` 和 `DEV_PROGRESS.md` 里写的是 14 个测试，当前仓库实际已经是 35 个测试
- `ENGINEERING.md` 有些示例还保留了更早期的 reducer 说明，但真实实现已经升级为 `merge_evidence()`

这类差异不是坏事，反而说明这是个还在演进的项目，不是静态样板。

---

## 4. 先建立一张学习脑图

目录层次如下：

```text
03_MULTI_AGENT/
├── app/          # 服务入口、生命周期、报告归档、多轮初始化
├── graph/        # LangGraph 主图、状态、路由
├── agents/       # planner / supervisor / researchers / reflector / writer
├── tools/        # HTTP 工具、MCP、RAG 适配、registry
├── prompts/      # 节点 prompt 模板
├── config/       # 设置、模型工厂、tracing
├── tests/        # 单测 + tutorial tests + 端到端离线测试
└── data/         # checkpoint / reports
```

最重要的依赖方向是：

```text
app -> graph -> agents -> tools
config 被所有层读取
prompts 被 agents 使用
```

这条依赖链非常重要，因为它说明项目不是把 API、agent、tool 全都搅在一个文件里，而是按照“接入层 -> 编排层 -> 节点层 -> 外部能力层”拆开的。

---

## 5. 用一条主流程带你看整个系统

如果你只记一条线，就记这条：

```text
HTTP 请求
  -> app/bootstrap.py 启动依赖
  -> app/api.py 接收 research_query
  -> graph/workflow.py 进入主图
  -> planner 生成计划并 interrupt
  -> resume 后 supervisor_route 扇出 Send
  -> researchers 并行取证
  -> merge_evidence fan-in 聚合
  -> reflector 判断是否补查
  -> writer 写报告并落盘
  -> turn_research 复用历史 evidence 做下一轮
```

接下来我们就按这条线，一个节点一个节点地学。

---

## 6. 先给你一张“流程 -> 代码 -> 测试 -> 学习目标”总表

这是全文最适合打印出来对照看的部分。

| 流程阶段 | 关键代码 | 对应测试 | 你应该学会什么 |
|---|---|---|---|
| 图能否搭起来 | `graph/workflow.py` | `tests/test_graph_skeleton.py` | 先确认主图节点齐全、编译正常 |
| Planner 中断恢复 | `agents/planner.py` | `tests/tutorial/test_05_interrupt_resume.py` | 理解 `interrupt()` / `Command(resume=...)` |
| Supervisor 并行扇出 | `graph/router.py` | `tests/tutorial/test_03_supervisor_send_fanout.py` | 理解 `Send` 如何把计划拆成并行 researcher 任务 |
| Researcher 工具降级 | `agents/_researcher_base.py` + `tools/registry.py` | `tests/tutorial/test_02_registry_degradation.py` | 理解主工具失败时如何 fallback |
| 节点级容错 | `agents/_safe.py` | `tests/tutorial/test_04_safe_node_decorator.py` | 理解为什么单个节点失败不能拖垮整张图 |
| 并行结果聚合 | `graph/state.py` | `tests/tutorial/test_01_state_reducer.py` | 理解 reducer 为什么要去重和排序 |
| Reflector 兜底 | `agents/reflector.py` | `tests/tutorial/test_06_reflector_hard_fallback.py` | 理解为什么最多只补查 3 轮 |
| Writer 收敛出报告 | `agents/writer.py` + `app/report_store.py` | `tests/test_end_to_end_offline.py` | 理解报告如何基于 evidence 生成并保存 |
| 全流程闭环 | `app/api.py` + `graph/workflow.py` | `tests/test_end_to_end_offline.py` | 把一次完整请求串起来 |

如果你时间很少，就按上表顺序读。

---

## 7. 第 0 步：启动阶段在准备什么

很多人会忽略启动阶段，但这个项目的工程味其实从启动就开始了。

### 7.1 看哪个文件

- `app/bootstrap.py`

### 7.2 它做了什么

启动时主要做四件事：

1. 初始化 `ToolRegistry`
2. 按顺序注册搜索工具和 MCP 工具
3. 初始化异步 SQLite checkpointer
4. 调 `build_graph(checkpointer=...)` 编译主图

关键点在这里：

- web 搜索链的注册顺序本身就是降级策略
- checkpointer 是服务级资源，不是请求级临时对象
- 图只编译一次，避免每次请求重复组装

### 7.3 这反映了什么设计思想

这说明项目作者不是把 LangGraph 当成一个随用随建的脚本工具，而是在按“后端服务”的方式组织生命周期。

### 7.4 实际工作里为什么这样选

- 工具客户端经常带连接、session、认证信息，适合在启动期统一初始化
- checkpointer 应该和图强绑定，否则 interrupt/resume 很容易错线程
- 如果每个请求都临时创建 graph，会放大延迟、连接成本和排查难度

### 7.5 你面试可以怎么讲

> 启动阶段做的不是简单 import，而是初始化整个研究流水线的基础设施，包括 tool registry、checkpoint 持久化和主图编译。这保证了后续每个请求都在同一套可恢复的执行环境上运行。

---

## 8. 第 1 步：HTTP 入口怎么把请求送进 LangGraph

这一步对应“用户发来一个研究问题，系统怎么开始跑”。

### 8.1 看哪个文件

- `app/api.py`

### 8.2 入口函数是谁

- `start_research()`
- `resume_research()`
- `turn_research()`

### 8.3 关键阅读点

`start_research()` 做的事情很克制：

1. 生成 `thread_id`
2. 构造最小初始 state
3. 调 `graph.ainvoke(payload, config)`
4. 判断是否命中 `__interrupt__`

初始 payload 只有：

```python
{
    "research_query": req.research_query,
    "audience": req.audience,
    "messages": [],
    "evidence": [],
}
```

这很值得学，因为它没有预先塞一堆默认字段，而是让后续节点按职责写状态。

### 8.4 为什么这样设计

这类设计有两个好处：

- 状态字段的“首次写入者”更清晰
- 测试更容易聚焦单节点行为，不会被大而全的初始状态污染

### 8.5 你要记住的工程点

- `thread_id` 不是装饰品，它是整个 checkpoint 恢复的主键
- `config={"configurable": {"thread_id": tid}}` 是 LangGraph 持久化恢复的关键
- API 层不做业务决策，只负责把请求送进图、把 interrupt 送回前端

### 8.6 代码精度带学笔记

先盯住 `app/api.py` 里的这几段关系：

```python
cfg = _config(tid)
result = await _invoke(payload, cfg)
interrupt_val = _extract_interrupt(result)
```

- `start_research()` 的主线不是业务判断，而是“构造 thread 维度 config -> 调图 -> 解析 interrupt”
- `_config()` 的返回值必须是 `{"configurable": {"thread_id": tid}}` 这个形状，否则 checkpointer 关联不到同一条执行链
- `_extract_interrupt()` 之所以要兼容 `list[Interrupt]` 和单个 `Interrupt`，是因为 LangGraph 返回结构并不是永远等形的普通 dict
- `resume_research()` 里最关键的不是路由，而是 `Command(resume={"plan": req.plan.model_dump()})`
- `turn_research()` 不是重开线程，而是在同一个 `thread_id` 下打一块 patch，让图继续在旧状态上运行

读这段代码时你要特别注意两个分支：

- `interrupt_val is None`：说明图已经收敛到 writer，可以直接返回 `final_report`
- `interrupt_val is not None`：说明图停在 Planner 的 HITL 断点，前端应该进入“计划确认”而不是“展示报告”

把这段和 `tests/test_end_to_end_offline.py` 一起看，你会明白 API 层的真实职责是“转运执行态”，不是“自己实现研究流程”。

---

## 9. 第 2 步：主图到底怎么连起来

### 9.1 看哪个文件

- `graph/workflow.py`

### 9.2 主图结构

```text
START
  -> planner
  -> supervisor
  -> researchers（条件扇出）
  -> reflector
  -> writer
  -> END
```

### 9.3 关键认识

这个项目最值得学的不是“有几个 agent”，而是：

- 图入口固定
- 条件分支显式
- 并行和回环都写在图里
- 收敛点清楚

### 9.4 对应测试

- `tests/test_graph_skeleton.py`

这个测试不是在测业务逻辑，而是在测“主图至少能被正确编译、节点至少没有缺失”。在真实工程里，这种骨架测试非常有价值，因为它能第一时间拦住 wiring 级别的破坏。

---

## 10. 第 3 步：Planner 是第一关键节点

现在正式进入 agent 流程。

### 10.1 看哪个文件

- `agents/planner.py`

### 10.2 这个节点的职责是什么

Planner 只做一件事：

> 把一个大的研究问题拆成结构化 `ResearchPlan`，然后暂停执行，等待用户确认。

### 10.3 代码里真正重要的三行

不是 prompt，而是这三件事：

1. `llm.with_structured_output(ResearchPlan, ...)`
2. `decision = interrupt({"phase": "plan_review", "plan": ...})`
3. `_coerce_plan(decision, fallback=plan)`

### 10.4 按流程理解这段代码

先生成结构化 plan：

- 输入：`research_query` + `audience`
- 输出：`ResearchPlan`

再中断：

- 不是 API 层猜测 planner 有没有结束
- 而是 Planner 节点自己声明“我现在要停下来让人确认”

最后恢复：

- 用户确认或修改后的内容通过 `Command(resume=...)` 回流
- 这个回流值会成为 `interrupt()` 的返回值
- `_coerce_plan()` 负责把不同序列化形态收敛成 `ResearchPlan`

### 10.5 Planner 会写哪些状态

- `plan`
- `plan_confirmed=True`
- `iteration=0`
- `revision_count=0`
- `current_node="planner"`
- `messages`

### 10.6 先看哪个测试

- `tests/tutorial/test_05_interrupt_resume.py`

这个测试非常重要，因为它直接演示了 HITL 的核心机制：

1. 第一次 `ainvoke()` 返回 `__interrupt__`
2. interrupt payload 里带有 planner 生成的 plan
3. 用户编辑后的 plan 通过 `Command(resume={"plan": edited})` 恢复
4. 恢复后 state 中实际采用的是“用户版本”

### 10.7 这个设计为什么高级

因为它把 HITL 做成了图内部能力，而不是前端补丁。

如果没有 `interrupt()`，你通常会写成：

- 先调一个 planner API
- 前端展示 plan
- 用户确认后再发另一个 execute API

这样的问题是：

- 状态会断裂
- 中断点无法自然持久化
- 恢复逻辑散落在 API 和前端

而现在的做法是：

- 中断点是 LangGraph 原生能力
- checkpoint 能存下中断现场
- 恢复后继续跑同一条图

### 10.8 实际工作里的选型与调优

为什么 Planner 用强模型：

- 任务拆解质量直接决定后续所有环节的成本和方向
- 子问题拆坏了，后面的搜索、反思、写作全都会跑偏

如果线上成本太高，Planner 的调优优先级通常是：

1. 限制子问题个数
2. 简化 `estimated_depth`
3. 对低复杂度 query 走更小模型
4. 对固定领域研究题增加模板化拆解

### 10.9 面试高频问答

Q：为什么计划确认要放在图里，而不是放在前端？

A：因为计划确认不是 UI 事件，而是研究流程的一部分。放进图里之后，才能和 checkpoint、thread_id、resume 一起组成完整的执行态。

### 10.10 代码精度带学笔记

先看 `planner_node()` 的真实执行顺序：

```python
structured = llm.with_structured_output(ResearchPlan, method="function_calling")
plan = await structured.ainvoke([...])
decision = interrupt({"phase": "plan_review", "plan": plan.model_dump()})
confirmed = _coerce_plan(decision, fallback=plan)
```

- 这里不是先 `interrupt()` 再生成计划，而是先拿到结构化 `ResearchPlan`，再把它作为 interrupt payload 交给前端
- `plan.model_dump()` 很关键，它把 Pydantic 模型变成适合跨边界传输的结构
- `_coerce_plan()` 的职责不是“再做一次计划”，而是把 `resume` 回来的多种形态统一成 `ResearchPlan`

读 `_coerce_plan()` 时要特别盯住这个顺序：

1. `decision is None` 时直接回退原 plan
2. `decision` 已经是 `ResearchPlan` 时直接使用
3. `decision` 是 dict 时先取 `decision.get("plan", decision)`
4. 只有 dict 里真的有 `sub_questions`，才尝试 `ResearchPlan.model_validate()`
5. 解析失败时记录 warning，并回退默认 plan

这说明 Planner 的容错重点不在 LLM，而在“恢复阶段的序列化兼容”。

`tests/tutorial/test_05_interrupt_resume.py` 里最值得对照看的断言是：

- 第一次调用必须有 `__interrupt__`
- 第二次 `Command(resume=...)` 后，`state.plan` 采用的是用户修改版本
- `plan_confirmed` 必须被写成 `True`

---

## 11. 第 4 步：Supervisor 很薄，但真正的调度很关键

### 11.1 看哪个文件

- `agents/supervisor.py`
- `graph/router.py`

### 11.2 最容易误解的点

很多人会以为 Supervisor 节点自己在做复杂调度，其实当前实现不是。

`agents/supervisor.py` 本身只做：

- `current_node="supervisor"`
- `iteration + 1`

真正决定下一步去哪的，是：

- `graph/router.py::supervisor_route()`

### 11.3 这个 route 函数有三种出口

1. 返回 `"planner"`
2. 返回 `"writer"`
3. 返回 `list[Send]`

分别对应：

- 计划未确认，回到 planner
- 已达最大 revision，直接收敛
- 计划已确认，需要并行调研

### 11.4 为什么 `list[Send]` 是关键

这说明 Supervisor 不是“自己跑完所有 researcher”，而是把任务描述拆成多个并行子任务交给 LangGraph。

每个 `Send` 本质上是在说：

> 把某个子问题，以某种 source_type 的 researcher 节点去执行。

### 11.5 `Send` 的并行粒度是什么

不是“按 agent 粗粒度并行”，而是：

> `sub_question × recommended_sources`

例如：

- `sq1` 需要 `web + academic`
- `sq2` 需要 `code`
- `sq3` 需要 `kb`

最终就会生成 4 个并行 researcher 任务。

这个粒度选得很好，因为：

- 扩新 source_type 很自然
- selective re-fanout 未来也好做
- 单个 source 出问题不会影响整个 sub_question

### 11.6 对应测试

- `tests/tutorial/test_03_supervisor_send_fanout.py`

这个测试保护的不是 prompt，而是路由契约：

- `plan_confirmed=False` 必须回 planner
- `revision_count >= 3` 必须直接 writer
- 首轮无 evidence 时必须 fan-out
- `Send.node` 映射必须正确
- `Send.arg` 里必须包含 `sub_question` 和 `research_query`
- `status="done"` 的子问题应该被跳过

### 11.7 为什么这是好设计

因为它把：

- 节点职责
- 路由决策
- 并行派发

拆成了不同层次，测试起来非常清晰。

如果把这些都塞进一个大节点里，会出现两个问题：

- 路由难测
- 节点逻辑和流程逻辑耦合

### 11.8 实际工作里的调优

如果项目进入生产环境，Supervisor/Router 层最值得做的调优是：

- 加 selective re-fanout，只补查缺失子问题
- 引入 provider 限流，避免 fan-out 后瞬时打爆外部 API
- 给 route 增加 query complexity 判断，简单问题直接 short path 到 writer

### 11.9 代码精度带学笔记

`supervisor_route()` 的代码精读顺序不要从底部开始，要按 guard clause 读：

```python
if not state.get("plan_confirmed"):
    return "planner"
if state.get("revision_count", 0) >= 3:
    return "writer"
if plan and not state.get("evidence") or state.get("next_action") == "need_more_research":
    return sends
return "writer"
```

- 第一个 `if` 是 HITL 恢复闸门，没确认计划就不允许进入 researcher
- 第二个 `if` 是成本闸门，超过轮数直接收敛
- 第三个 `if` 才是业务调度闸门，决定是否 fan-out

这里最容易看错的是这个条件：

```python
if plan and not state.get("evidence") or state.get("next_action") == "need_more_research":
```

按 Python 运算优先级，它等价于：

```python
if (plan and not state.get("evidence")) or state.get("next_action") == "need_more_research":
```

也就是说，只要 Reflector 给了 `need_more_research`，即使当前已经有 evidence，也会重新 fan-out。

再看 `Send(...)` 的 arg：

```python
{
    "sub_question": sq,
    "research_query": state.get("research_query", ""),
}
```

- `sub_question` 给 researcher 做本地任务输入
- `research_query` 作为兜底上下文，避免 payload 不完整时完全丢失查询语义

`tests/tutorial/test_03_supervisor_send_fanout.py` 实际保护的就是这几个分支顺序和 payload 契约。

---

## 12. 第 5 步：Researcher 节点为什么说“简单但值钱”

### 12.1 看哪些文件

- `agents/_researcher_base.py`
- `agents/researcher_web.py`
- `agents/researcher_academic.py`
- `agents/researcher_code.py`
- `agents/researcher_kb.py`
- `tools/registry.py`

### 12.2 Researcher 的职责边界

Researcher 只做一件事：

> 从某类数据源里拿证据，然后统一映射成 `Evidence`。

它不负责：

- 大规模总结
- 最终写作
- 总流程决策

这就是为什么共同逻辑被抽到了 `_researcher_base.py`。

### 12.3 先理解 `run_research_chain()`

这个函数的逻辑非常务实：

1. 从 `registry.get_chain(source_type)` 取一条工具链
2. 顺序调用每个 tool
3. 某个 tool 返回非空结果就短路
4. 某个 tool 异常则记日志并降级
5. 全部失败则返回空列表

这就是标准的 fallback chain。

### 12.4 为什么 `ToolRegistry` 值钱

`ToolRegistry` 的价值不在“存了几个工具”，而在“定义了业务优先级”。

比如 web 搜索链是按启动顺序注册的：

- Tavily
- external MCP Brave
- DashScope Search

顺序本身就是策略。

### 12.5 对应测试

- `tests/tutorial/test_02_registry_degradation.py`

这个测试保护了四件事：

1. 主工具异常时必须调用下一个
2. 中间工具返回空结果时必须继续降级
3. 所有工具都失败时返回 `[]`
4. 第一个工具命中后必须短路，不再浪费成本

### 12.6 单个 researcher 节点本身长什么样

以 `agents/researcher_web.py` 为例，它只是：

1. 从 payload 里解析 `sub_question`
2. 调 `run_research_chain(source_type="web", ...)`
3. 返回 `{"evidence": evidence, "messages": ...}`

这就是好的节点设计：

- 节点本身薄
- 业务共性抽到 base
- 外部依赖通过 registry 注入

### 12.7 为什么没有在 researcher 阶段大量用 LLM

这是非常现实的成本控制。

Researcher 阶段最重要的是：

- 广覆盖
- 快拿证据
- 稳定拿到结构化来源

如果每条搜索结果都再让 LLM 提炼一次，会带来：

- token 成本膨胀
- 时延放大
- 失败点增加

所以当前实现优先“取证”，而不是“先润色证据”。

### 12.8 实际工作里的技术选型与调优

Researcher 层的工作重点通常不是模型，而是：

- provider 可靠性
- timeout 策略
- fallback 顺序
- 结果标准化
- snippet 截断

如果我要继续做这层调优，我会优先加：

1. provider 成功率埋点
2. 429 / timeout 分类降级
3. source_url 级缓存
4. query 相似度缓存
5. 单 provider 并发上限

### 12.9 面试高频问答

Q：为什么 Researcher 不直接让大模型总结成结论？

A：因为 Researcher 层的任务是建立证据池，不是出最终观点。过早总结既增加成本，也容易把原始证据损失掉。

### 12.10 代码精度带学笔记

`run_research_chain()` 最应该精读的不是 for 循环本身，而是“短路顺序”：

```python
chain = registry.get_chain(source_type) if registry else []
for tool in chain:
    try:
        results = await asyncio.wait_for(tool.search(query, top_k=top_k), timeout=45)
    except Exception:
        continue
    if results:
        return _to_evidence(results, ...)
return []
```

- 先拿 `chain`，说明 Researcher 不认识具体 provider，只认识 source_type
- 每个 tool 调用外面套了 `asyncio.wait_for(..., timeout=45)`，这说明超时也被视为正常降级场景
- `except Exception: continue` 表示某个 provider 挂了不影响后续 provider
- `if results: return ...` 表示首个命中的 provider 会短路，后面 provider 不再浪费配额

再看 `_to_evidence()`：

- `url = r.get("source_url") or ""` 为空会直接跳过
- `snippet` 被截到 2000 字符，这是在控制后续 writer/reflector 的输入成本
- `relevance_score` 被强制转成 `float`，这是为了保证 reducer 排序时类型稳定

`extract_sq_and_query()` 也值得单独看，因为它兼容了：

- Pydantic `SubQuestion`
- dict 形态的 sub_question
- payload 缺字段时退回 `research_query`

`tests/tutorial/test_02_registry_degradation.py` 不是在测搜索结果质量，而是在测这条 fallback chain 的执行语义是否稳定。

---

## 13. 第 6 步：safe_node 为什么是生产级小细节

### 13.1 看哪个文件

- `agents/_safe.py`

### 13.2 它做了什么

`safe_node` 把节点异常转换成：

```python
{
    "evidence": [],
    "messages": [AIMessage(content="[skip] ...")]
}
```

也就是说，单个 researcher 失败时：

- 主图不崩
- 其他并行 researcher 还能继续
- reflector 还可以基于“不足的 evidence”决定是否补查

### 13.3 对应测试

- `tests/tutorial/test_04_safe_node_decorator.py`

这个测试验证了：

- 异常会被转换成合规返回值
- message 里保留函数名和错误文本
- 正常返回值不受影响
- `functools.wraps` 保留原函数名

### 13.4 为什么这很重要

很多 demo 项目只要一个工具报错，整张图就直接炸掉。这个项目选择的是软降级：

- 不让某个外部 API 成为整条研究链路的单点故障
- 保留系统继续运行的能力
- 把失败变成“证据不足”，而不是“系统不可用”

这就是工程化和 demo 的差别。

### 13.5 代码精度带学笔记

`safe_node` 的实现很短，但要精读这两个点：

```python
@functools.wraps(fn)
async def wrapper(state, *args, **kwargs):
    try:
        return await fn(state, *args, **kwargs)
    except Exception as e:
        return {"evidence": [], "messages": [AIMessage(...)]}
```

- `functools.wraps(fn)` 不是装饰，是为了保住原函数名，调试和 tracing 才不会全变成 `wrapper`
- `return {"evidence": [], "messages": ...}` 说明异常后的返回值仍然遵守节点契约，而不是返回一个 ad-hoc 错误对象

这里最容易忽略的一点是：

- 它没有吞掉异常后返回 `None`
- 也没有抛出 HTTP 级错误
- 它返回的是“图还能继续流转”的最小合法状态

`tests/tutorial/test_04_safe_node_decorator.py` 正在保护这件事：

- 返回值必须仍是 dict
- `evidence` 必须是空列表而不是缺字段
- message 内容里要带函数名和异常文本，方便定位故障

---

## 14. 第 7 步：并行 fan-in 真正落在 reducer 上

### 14.1 看哪个文件

- `graph/state.py`

### 14.2 为什么 `ResearchState` 是最关键的数据结构

因为这个项目真正共享的不是聊天历史字符串，而是结构化研究状态。

最关键字段按阶段分组如下：

输入：

- `research_query`
- `audience`

计划：

- `plan`
- `plan_confirmed`

调研：

- `evidence`
- `revision_count`

反思：

- `coverage_by_subq`
- `missing_aspects`
- `next_action`
- `additional_queries`

输出：

- `final_report`
- `citations`
- `report_path`

流程辅助：

- `messages`
- `current_node`
- `iteration`

### 14.3 为什么 `evidence` 不能简单拼接

当前真实实现用的是：

- `merge_evidence()`

不是早期示例里那种 `operator.add`

它做三件事：

1. 按 `source_url` 去重
2. 同 URL 只保留 `relevance_score` 更高的一条
3. 最终按相关度倒序

### 14.4 对应测试

- `tests/tutorial/test_01_state_reducer.py`

这个测试非常适合入门，因为它把 reducer 的真实语义讲得很清楚：

- 重复 URL 只保留高分项
- dict / Pydantic / None 混合输入都能处理
- 空 URL 会被过滤
- 输出顺序按分数倒序

### 14.5 这里体现的设计思想

并行不是难点，fan-in 才是难点。

如果没有 reducer，你最终会遇到这些问题：

- evidence 重复堆积
- writer 输入长度膨胀
- reflector 误判覆盖度
- 同一个观点因为多个来源重复采样而被“虚假放大”

所以 `merge_evidence()` 本质上是质量控制，不只是代码清洁。

### 14.6 实际工作里的调优

这层最常见的继续调优方向是：

- 把 URL 级去重升级为“URL + 语义近似”双重去重
- 给 evidence 加 freshness / source_authority 等质量权重
- 对同一 sub_question 做覆盖面采样，而不是只按 relevance 排序

### 14.7 代码精度带学笔记

读 `merge_evidence()` 时，建议按“归一化 -> 去重 -> 排序”三段看：

```python
pool = list(old or []) + list(new or [])
for raw in pool:
    d = _to_dict(raw)
    url = d.get("source_url") or ""
    ...
merged.sort(key=lambda e: -float(e.relevance_score or 0.0))
```

第一段，归一化：

- `_to_dict()` 兼容 Pydantic 模型和 dict
- 遇到非 Evidence-like 对象直接 `TypeError`，上层会跳过

第二段，去重：

- key 不是 `sub_question_id`，而是 `source_url`
- 同 URL 命中多次时，不做 merge list，而是保留 `relevance_score` 更高的版本

第三段，排序：

- 排序发生在去重之后
- 排序 key 是 `-float(e.relevance_score or 0.0)`，保证空值也不会炸

这意味着 reducer 的核心目标不是“保留所有证据”，而是“保留最值得进入后续推理的证据”。

`tests/tutorial/test_01_state_reducer.py` 里四个测试基本一一对应四种风险：

- 重复 URL 覆盖风险
- 输入形态不一致风险
- 空 URL 污染风险
- 输出顺序不稳定风险

---

## 15. 第 8 步：Reflector 控制的是质量边界和成本边界

### 15.1 看哪个文件

- `agents/reflector.py`
- `graph/router.py::reflector_route()`

### 15.2 Reflector 的职责

Reflector 不是为了“显得 agent 很高级”，它有两个非常现实的职责：

1. 判断证据是否足够
2. 控制补查回路不要失控

### 15.3 它具体怎么做

Reflector 先把：

- `plan`
- `evidence`

压缩成摘要，再让强模型输出结构化 `ReflectionResult`，包括：

- `coverage_by_subq`
- `missing_aspects`
- `next_action`
- `additional_queries`

然后把这些结果写回 state。

### 15.4 最关键的不是 LLM，而是硬兜底

当前代码里有一条非常重要的逻辑：

```python
if rc >= MAX_REVISION:
    return {"next_action": "force_complete", ...}
```

也就是说：

- 第 3 轮时直接强制收敛
- 不再继续调 LLM 做“无限反思”

### 15.5 对应测试

- `tests/tutorial/test_06_reflector_hard_fallback.py`

这个测试保护的是项目里最像生产规则的一条约束：

- `revision_count=2` 时进入 reflector，不应该再调用 LLM
- 应该直接返回 `force_complete`
- 首轮时则必须调用 LLM 做正常评分

### 15.6 为什么必须有这条边界

如果没有反思上限，系统会出现经典问题：

- 一直觉得证据不够
- 一直 fan-out 补查
- 成本和时延不断放大
- 用户迟迟拿不到结果

研究任务天然存在收益递减，所以“足够好并及时返回”通常比“无限追求完美”更对。

### 15.7 实际工作里的调优

Reflector 层最重要的调优是“少但准”：

- 让它只在证据不足或冲突明显时工作
- 简化 evidence 摘要输入，避免 token 浪费
- 对简单 query 甚至可以直接跳过 reflector

面试里如果被问“为什么不让它一直反思到满意”，标准回答就是：

> 因为真实系统有 SLA 和成本边界，反思循环必须显式设上限。

### 15.8 代码精度带学笔记

`reflector_node()` 建议你先看顶部这两行：

```python
rc = state.get("revision_count", 0) + 1
if rc >= MAX_REVISION:
    return {"next_action": "force_complete", ...}
```

- `rc` 不是直接读旧值，而是先 `+1`，因为当前这次 reflector 调用本身就算一次 revision
- 硬兜底分支发生在 LLM 调用之前，所以第 3 轮时根本不会创建 structured output，也不会消耗模型额度

如果没走硬兜底，后面才会进入：

```python
structured = llm.with_structured_output(ReflectionResult, method="function_calling")
result = await structured.ainvoke([...])
```

再注意 evidence 摘要的生成方式：

- 先按 `sub_question_id` 分桶
- 每个桶最多取前 5 条
- 每条 snippet 只取前 200 字

这说明 Reflector 的输入不是“全量 evidence 原文”，而是压缩过的 coverage 视图。

`tests/tutorial/test_06_reflector_hard_fallback.py` 要盯住两个断言：

- `revision_count=2` 时，`spy.calls == 0`
- `revision_count=0` 时，`spy.calls == 1`

这两个断言共同证明：硬兜底不是 prompt 要求，而是代码路径保证。

---

## 16. 第 9 步：Writer 是“LLM + 确定性后处理”的组合

### 16.1 看哪些文件

- `agents/writer.py`
- `app/report_store.py`

### 16.2 Writer 的职责

Writer 的任务不是“从 0 编故事”，而是：

> 基于当前 plan 和 evidence，生成结构化 Markdown 报告，并且把引用体系收敛到可保存的格式。

### 16.3 代码里最值得学的点

Writer 先做了一层确定性预处理：

- 给 evidence 编号
- 生成 `numbered_evidence`
- 让模型按 `[^N]` 引用

然后再做一层确定性后处理：

- 由后端自己生成 `citations`
- 如果正文缺少引用章节，就自动补 `## 引用`

这说明 Writer 不是单纯靠 prompt 撑住，而是：

> 模型写正文，系统掌握事实结构和落盘逻辑。

### 16.4 报告如何落盘

`app/report_store.py` 会按：

```text
{timestamp}_{slug(query)}_{thread_id}.md
```

的格式写入文件。

这个命名方式很好，因为它同时满足：

- 人可读
- 可按时间排序
- 可按 thread_id 回查

### 16.5 对应测试

最直接看：

- `tests/test_end_to_end_offline.py`

这个测试虽然不是专门只测 writer，但它会把整条链路跑通到最终报告生成，因此能验证：

- planner interrupt
- resume 后 researcher 拿到 evidence
- reflector 收敛
- writer 产出 `final_report`

### 16.6 为什么这一步是高价值设计

很多 Agent 项目写报告时只返回一段字符串，后续：

- 不好归档
- 不好复盘
- 不好做 report list
- 不好作为下一轮上下文复用

这里把报告保存成文件，并记录 `report_path`，工程可用性会高很多。

### 16.7 实际工作里的调优

Writer 层最值得做的优化通常是：

1. 控制 evidence 输入数量
2. 控制单条 snippet 长度
3. 区分 quick/standard/deep 报告模板
4. 对引用做更严格的格式校验
5. 对报告内容做 post-check，比如缺结论、缺引用时重试

### 16.8 代码精度带学笔记

`writer_node()` 的精读重点是“先构造 evidence 视图，再调模型，再补结构”。

第一段：

```python
for i, ev in enumerate(evidence, 1):
    numbered.append(f"[{i}] ...")
    citations.append(Citation(idx=i, source_url=ev.source_url, title=title))
```

- evidence 编号从 1 开始，对应脚注习惯
- `Citation` 不是让模型生成，而是后端从 evidence 直接构造
- `title` 来自 snippet 前 80 个字符，说明这里优先追求可回看性，不追求完美标题抽取

第二段：

```python
resp = await llm.ainvoke([...])
report_md = (resp.content if hasattr(resp, "content") else str(resp)).strip()
```

- 这里真实实现是 `ainvoke()`，不是旧注释里说的 sync invoke
- 结果读取兼容 `Message.content` 和普通字符串

第三段：

```python
if not _has_citation_section(report_md) and citations:
    report_md += "\n\n## 引用" + ...
```

- 这说明“有引用章节”是系统级要求，不完全信任模型遵守
- `_has_citation_section()` 用的是正则，不是简单字符串 contains，因此兼容 `引用`、`参考文献`、`references`

第四段：

```python
thread_id = (config or {}).get("configurable", {}).get("thread_id", "unknown")
path = report_store.save(query, thread_id, report_md)
```

- Writer 落盘时依赖 graph config 里的 `thread_id`
- 如果没有 config，也不会崩，而是退回 `"unknown"`，这是典型的防御式写法

把这段和 `app/report_store.py` 一起看，你会更容易明白报告不是普通字符串，而是一个可回查资产。

---

## 17. 第 10 步：多轮追问为什么不是重开一次会话

### 17.1 看哪些文件

- `app/api.py::turn_research()`
- `app/turn_init.py`

### 17.2 它怎么做

多轮追问时不会新建 thread，而是：

1. 继续使用原来的 `thread_id`
2. 调 `reset_per_turn({}, new_query)`
3. 保留历史 `evidence` / `plan` / `final_report`
4. 重置 `revision_count` / `iteration` / `missing_aspects` 等易变字段
5. 显式把 `plan_confirmed=False`，再回到 planner

### 17.3 为什么这是好设计

这说明系统把状态分成了两类：

- 事实资产：应该跨轮复用
- 过程变量：应该每轮重算

如果不做这层区分，就会出现两个极端：

- 要么什么都不保留，追问退化成重跑
- 要么什么都保留，上一轮过程变量污染下一轮

### 17.4 实际工作里的原则

一句话概括就是：

> 前端保存展示态，后端保存执行态。

这个项目就是按这个原则做的。

### 17.5 代码精度带学笔记

`turn_research()` 的精读重点是这两行：

```python
patch = reset_per_turn({}, req.research_query)
patch["plan_confirmed"] = False
```

- `reset_per_turn()` 返回的是 patch，不是整份新 state
- 这意味着旧线程里原有的 `evidence`、`plan`、`final_report` 仍然由 checkpoint 保留
- 新 patch 只负责覆写“这一轮必须变”的字段

再看 `reset_per_turn()` 本身：

```python
for k in PER_TURN_RESET_FIELDS:
    patch[k] = 0 if k in {"revision_count", "iteration"} else None
patch["coverage_by_subq"] = {}
patch["missing_aspects"] = []
```

- 计数字段重置成 `0`
- 流程指针类字段重置成 `None`
- 集合类字段重置成空 dict / 空 list

这里体现的是“按字段语义重置”，不是粗暴地全部置空。

`patch["plan_confirmed"] = False` 也很关键：

- 它逼着图重新经过 Planner
- 但 Planner 此时是在历史 evidence 已经存在的线程上下文里重新决策
- 所以多轮追问不是“纯追加”，而是“带着历史资产重规划”

---

## 18. 用一次“完整请求”把所有 agent 串起来

现在你可以把整个流程口述成下面这样：

1. 用户调用 `/research` 发起研究请求
2. `app/api.py` 创建 `thread_id`，用最小 state 调 `graph.ainvoke()`
3. 主图进入 `planner`
4. `planner` 生成结构化 `ResearchPlan`，然后 `interrupt()` 等待人工确认
5. 前端把 plan 给用户编辑，后端用 `Command(resume={"plan": ...})` 恢复
6. `supervisor_route()` 根据已确认计划生成 `list[Send]`
7. 多个 `researcher` 节点并行从 web / academic / code / kb 取证
8. 并行结果通过 `merge_evidence()` 自动 fan-in 聚合
9. `reflector` 判断证据是否充分，不够就补查，最多 3 轮
10. `writer` 基于 evidence 生成 Markdown 报告，并保存到本地
11. 后续追问通过 `/research/{thread_id}/turn` 复用历史 evidence 再进入下一轮

如果你能把上面这 11 步按顺序讲清楚，这个项目你就已经不是“看懂”，而是“能讲懂”了。

---

## 19. 为什么这个项目适合拿来讲 LangGraph

因为它覆盖了 LangGraph 最有代表性的几个能力，而且不是表面用法：

- `StateGraph`
- `TypedDict` state
- reducer
- `Send`
- `interrupt()`
- `Command(resume=...)`
- checkpointer
- 多轮 thread 恢复

更重要的是，它没有把 LangGraph 当成“多 agent 话术框架”，而是当成：

> 工作流编排引擎 + 状态机 + 智能节点容器

这才是更像生产系统的用法。

---

## 20. 结合实际工作讲技术选型

这一节是面试里最容易拉开差距的部分。

### 20.1 为什么选 LangGraph

不是因为它“新”，而是因为这个场景需要：

- 显式状态
- 条件路由
- 并行 fan-out / fan-in
- 中断恢复
- 持久化 replay

如果只是做一个 agent demo，别的框架也能做。

但如果要做一个：

- 有流程阶段
- 能中断
- 能恢复
- 能补查
- 能跨轮复用状态

的研究流水线，LangGraph 更合适。

### 20.2 为什么选 FastAPI

因为这个项目的 API 需求很典型：

- 异步请求
- Pydantic schema
- 生命周期管理
- 后续可扩展 SSE

FastAPI 在这里是低摩擦选择。

### 20.3 为什么当前阶段选 SQLite checkpointer

当前目标是：

- 本地可跑
- 作品集易部署
- 单机场景下能稳定恢复 thread

SQLite 足够轻量，而且门槛低。

如果进入多实例生产，再考虑：

- PostgreSQL
- 更中心化的持久化后端
- 更完善的 checkpoint 管理

### 20.4 为什么选 Qwen + DashScope

这是个非常现实的工程取舍：

- 中文能力强
- 国内环境可达性好
- 成本更可控
- OpenAI compatible endpoint 便于接 LangChain

更妙的是这个项目没有把 DashScope 只当成一个模型供应商，而是拆成：

- LLM 能力
- 搜索兜底能力

这就是按能力分工选型，而不是“一个平台包打天下”。

### 20.5 为什么要统一 `SearchTool` 协议

因为上层节点应该关心：

- 我要 web 证据
- 我要 academic 证据

而不应该关心：

- 你是 HTTP API
- 你是 MCP tool
- 你是 RAG retriever

协议统一之后：

- fallback 好做
- provider 替换容易
- 测试更容易打桩

### 20.6 为什么 internal MCP 也值得讲

项目不仅会“消费外部工具”，还会“把自己变成工具”。

这意味着：

- 这个系统不是封闭应用
- 它可以成为其他 agent 体系中的一个能力节点

这在实际工作里非常有价值，因为系统一旦能工具化，就更容易复用。

---

## 21. 结合实际工作讲调优

### 21.1 模型调优

当前强模型主要放在：

- Planner
- Reflector
- Writer

这很合理，因为这三处最吃推理和表达能力。

如果要降本，优先级通常是：

1. 减少 Planner 拆出的子问题数量
2. 控制 Writer 输入的 evidence 数量
3. 缩短 snippet 长度
4. 简单问题跳过 Reflector
5. 按 query complexity 分流模型档位

### 21.2 搜索链调优

真实世界里，外部搜索最常见的问题不是“搜不到”，而是：

- 配额不稳
- 区域网络波动
- 接口超时
- 结果格式不一致

所以这个项目的高价值点不是某个 provider，而是：

> 它承认单一 provider 不可靠，因此从一开始就按降级链来设计。

如果继续往生产做，我会补：

- provider 成功率监控
- 429 专项退避
- 按 query 类型做 provider routing
- 命中缓存

### 21.3 并发调优

现在项目已经具备 fan-out，但还没有统一的限流层。

这是当前阶段可以接受的，因为：

- 先证明流程跑通
- 再做资源调优

如果真要上线，我会加：

- 全局 semaphore
- 单 provider 并发上限
- 同一 query 的短期缓存
- 同一 URL 的重复请求抑制

### 21.4 状态调优

`merge_evidence()` 的去重不是细节，而是状态质量控制。

如果 evidence 不去重，会直接带来：

- writer token 膨胀
- reflector 高估覆盖度
- 报告里重复引用

所以 reducer 的设计其实是在调质量和成本。

### 21.5 可靠性调优

`safe_node` 和 fallback chain 的组合很像真实线上系统：

- 单点故障不拖垮主链路
- 出错时尽量转为“降级”而不是“宕机”

这类设计比单纯追求 happy path 漂亮重要得多。

---

## 22. 面试里最值得讲的设计思想

你可以把这个项目提炼成下面 5 个设计思想。

### 22.1 流程显式化

不是让 agent 自己随便想下一步，而是：

- 节点显式
- 状态显式
- 路由显式

### 22.2 重推理节点少而精

只有 Planner、Reflector、Writer 真正大量使用强模型。

其他地方优先走：

- 规则
- 协议
- 工具
- 状态

### 22.3 状态是第一公民

不是靠对话 history string 维持上下文，而是靠结构化 `ResearchState`。

### 22.4 并行之后必须设计 fan-in

很多项目会讲 fan-out，但很少认真处理 fan-in。这个项目通过 reducer 明确解决了并行聚合问题。

### 22.5 把用户参与设计进流程

HITL 不是前端插个确认框，而是图里的一个正式阶段。

---

## 23. 面试高频问题与参考回答

### Q1：为什么要做多 Agent，不是一个大模型直接写报告？

A：因为这个任务天然分阶段，先拆计划、再搜证据、再判断覆盖、最后写报告。单模型一次性完成会让证据链、状态恢复和用户纠偏都很弱。

### Q2：为什么计划确认一定要用 `interrupt()`？

A：因为它能把中断点纳入执行态，让 checkpoint 和 resume 成为自然的一部分，而不是 API 和前端之间的手写协议。

### Q3：为什么 Supervisor 节点这么薄？

A：这是刻意设计。当前版本把复杂调度尽量下沉到显式 route，换来更强的可预测性和可测试性。

### Q4：为什么 Researcher 不直接总结成最终结论？

A：Researcher 的职责是建立证据池。过早总结既增成本，也损失原始证据的可复用性。

### Q5：为什么 reducer 不直接做 `operator.add`？

A：因为真实搜索会出现重复 URL。简单拼接会导致重复证据污染 writer 和 reflector。

### Q6：为什么 Reflector 要限制轮数？

A：因为研究任务有收益递减，系统必须有成本和时延边界，不能为了“更完整”无限循环。

### Q7：为什么工具层要统一协议？

A：因为上层应该依赖“能力类型”而不是具体厂商。这样 provider 替换、fallback、打桩测试都更容易。

### Q8：为什么要保留 report_path 并写本地文件？

A：因为报告不是瞬时字符串，而是可复盘资产。落盘后更适合归档、回查、展示和多轮复用。

### Q9：为什么这个项目更像工程项目而不是 demo？

A：因为它有 checkpoint、interrupt/resume、fallback、safe_node、report store、multi-turn 和一组保护核心机制的测试。

### Q10：如果继续做下一步，你会优先做什么？

A：我会优先补三件事：SSE 流式事件、selective re-fanout、可观测性与评测。

---

## 附：代码精度学习索引表

如果你准备边看文档边开代码，这张表最适合当导航。

| 知识点 | 先看文件 | 再看测试 | 你要盯住的代码细节 |
|---|---|---|---|
| thread_id 如何贯穿整条图 | `app/api.py` | `tests/test_end_to_end_offline.py` | `_config()` 产出的 `configurable.thread_id` 形状 |
| interrupt/resume 的真实语义 | `agents/planner.py` | `tests/tutorial/test_05_interrupt_resume.py` | `interrupt(...)` 的返回值如何变成 `_coerce_plan()` 的输入 |
| 为什么 Supervisor 本身很薄 | `agents/supervisor.py` + `graph/router.py` | `tests/tutorial/test_03_supervisor_send_fanout.py` | 路由逻辑不在节点里，而在 `supervisor_route()` |
| fan-out 的真实粒度 | `graph/router.py` | `tests/tutorial/test_03_supervisor_send_fanout.py` | `sub_question × recommended_sources -> list[Send]` |
| fallback chain 如何短路 | `agents/_researcher_base.py` | `tests/tutorial/test_02_registry_degradation.py` | `if results: return _to_evidence(...)` |
| 节点异常为什么不该中断主图 | `agents/_safe.py` | `tests/tutorial/test_04_safe_node_decorator.py` | 异常后返回最小合法状态而不是 `None` |
| reducer 为什么不能只拼接 list | `graph/state.py` | `tests/tutorial/test_01_state_reducer.py` | URL 去重、高分覆盖、倒序排序 |
| Reflector 为什么第 3 轮不再调模型 | `agents/reflector.py` | `tests/tutorial/test_06_reflector_hard_fallback.py` | `rc = old + 1` 后立即走硬兜底 |
| Writer 为什么是“LLM + 后处理” | `agents/writer.py` | `tests/test_end_to_end_offline.py` | evidence 编号、citation 后端生成、缺章节自动补 |
| report_path 为什么能支持回查 | `app/report_store.py` | `tests/test_end_to_end_offline.py` | 文件名里包含 `timestamp + slug + thread_id` |
| 多轮追问为什么不是重开线程 | `app/api.py` + `app/turn_init.py` | `tests/test_end_to_end_offline.py` | `reset_per_turn()` 只打 patch，不清空历史资产 |

你可以按这张表做一个高强度精读：

1. 先打开“先看文件”
2. 只盯“你要盯住的代码细节”
3. 然后立刻去看对应测试断言
4. 最后再回到正文理解它在全流程中的位置

这样读，理解速度会明显快于“先把所有代码读一遍再说”。

---

## 24. 最适合你的精读顺序

如果你准备自己二刷这个项目，我建议按下面顺序：

1. `tests/tutorial/test_01_state_reducer.py`
2. `graph/state.py`
3. `tests/tutorial/test_05_interrupt_resume.py`
4. `agents/planner.py`
5. `tests/tutorial/test_03_supervisor_send_fanout.py`
6. `graph/router.py`
7. `tests/tutorial/test_02_registry_degradation.py`
8. `agents/_researcher_base.py`
9. `tests/tutorial/test_04_safe_node_decorator.py`
10. `agents/_safe.py`
11. `tests/tutorial/test_06_reflector_hard_fallback.py`
12. `agents/reflector.py`
13. `tests/test_end_to_end_offline.py`
14. `agents/writer.py`
15. `app/api.py`
16. `app/bootstrap.py`

这样读的好处是：

- 先搞懂状态和 reducer
- 再搞懂 interrupt/resume
- 再搞懂并行扇出和工具降级
- 最后再把整条链路串起来

这比从 `api.py` 顺着点开快得多。

---

## 25. 你可以直接照着说的一段项目总结

> 我做了一个基于 LangGraph 的多 Agent 深度研究系统，主流程是 Planner 先把研究问题拆成结构化 plan，并通过 interrupt 进入人工确认；用户确认后，Supervisor 基于 plan 通过 Send 把子问题按 source_type 并行派发给多个 Researcher；Researcher 通过 ToolRegistry 统一接不同 provider，并走 fallback chain 收集证据；并行 evidence 通过 reducer 做去重和排序；Reflector 评估证据覆盖度并控制补查轮数；Writer 最后基于 evidence 生成带引用的 Markdown 报告并落盘，同时整个流程通过 SQLite checkpointer 支持 thread 级恢复和多轮追问。

这段话的重点是：

- 说清流程
- 说清机制
- 说清工程化

不要只说“我做了 7 个 agent”，那种表达信息量太低。

---

## 26. 当前项目的真实验证结论

基于当前仓库真实代码，可以确认：

- `PROJECT_SCENARIO.md` 是真实 PRD 文件名
- 主图节点齐全，可成功编译
- interrupt/resume 机制有 tutorial test 保护
- `Send` fan-out 路由有 tutorial test 保护
- reducer 去重逻辑有 tutorial test 保护
- fallback chain 有 tutorial test 保护
- reflector 强制收敛有 tutorial test 保护
- 端到端离线闭环测试可以跑通到 writer

因此你在学习和面试里，应该优先讲：

- 已实现的后端闭环
- 状态设计
- 流程编排
- 容错与调优

而不是把还没落地的 SSE/UI 当成已完成能力去讲。

---

## 27. 一句话收束

这个项目最值得你学的，不是“怎么堆多几个 agent”，而是：

> 如何把一个复杂研究任务拆成一条可中断、可恢复、可并行、可降级、可验证的工程化流水线。
