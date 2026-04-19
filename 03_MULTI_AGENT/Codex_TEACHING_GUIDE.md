# InsightLoop 项目教学文档

本文基于 `03_MULTI_AGENT` 当前代码、`PROJECT_SCENARIO.md`、`ENGINEERING.md`、`DEV_PROGRESS.md`、测试集和样例报告整理。目标不是重复设计文档，而是带你按真实执行流程读懂代码、理解设计取舍，并能把它讲成一段有工程说服力的项目经历。

---

## 1. 先给结论：这个项目到底是什么

`InsightLoop` 是一个面向“深度研究”场景的多 Agent 系统。它解决的不是普通问答，而是下面这类任务：

- 题目复杂，需要先拆问题，再找资料，再综合写结论
- 信息来源不止一种，要同时查网页、论文、GitHub、私有知识库
- 用户不希望系统闷头跑错方向，所以中途要能确认计划
- 第一轮产出报告后，还要支持继续追问，而不是重新开一次新对话

从工程角度看，它本质上是：

1. `LangGraph` 负责流程编排与状态机
2. `FastAPI` 提供服务入口
3. `SQLite checkpointer` 负责跨轮持久化
4. 多种工具统一走 `SearchTool` 协议
5. `Planner / Reflector / Writer` 负责高价值推理
6. `Researcher` 负责并行拉取证据

一句话理解：

这是一个“显式状态机驱动的多 Agent 研究流水线”，不是一个“让大模型自己随便决定下一步”的黑盒聊天系统。

---

## 2. 当前项目完成到什么程度

先把“设计稿”和“实际代码”分开看，否则容易读乱。

当前代码实际完成的是：

- Planner + HITL 中断恢复
- Supervisor 路由 + `Send` 并行扇出
- 4 个 Researcher
- Reflector 反思补查
- Writer 报告生成与归档
- FastAPI 基础接口
- External MCP client
- Internal MCP server
- 多轮状态恢复
- 离线测试闭环

还没完成的是：

- SSE 流式事件接口
- Streamlit UI
- Docker Compose
- LangSmith 真正接线
- 完整评测系统

也就是说，这个项目当前是一个“后端闭环 + 工程骨架完整”的版本，适合作为面试作品集里的系统设计与工程实现项目。

几个你读代码时必须知道的“文档和实现差异”：

- `ENGINEERING.md` 里写了 `/research/{tid}/stream` SSE，但 `app/api.py` 里还没实现
- 文档里说 Supervisor 有较多调度智能，但当前代码里真正决定扇出的是 `graph/router.py::supervisor_route()`，`agents/supervisor.py` 本身很薄
- `agents/writer.py` 的注释还写着“sync invoke”，但实际实现已经是 `await llm.ainvoke(...)`
- `README.md`、`DEV_PROGRESS.md` 里写“14 单测全绿”，实际现在测试是 `35 passed`

这几处差异本身很正常，说明项目正在迭代，不是静态模板。

---

## 3. 建议的学习顺序

如果你想最快读懂，不要先从 API 开始逐文件硬啃。建议按下面顺序：

1. 先看 `PROJECT_SCENARIO.md`
2. 再看 `ENGINEERING.md`
3. 再看这几个测试
4. 再回头读核心代码

最适合入门的测试顺序：

1. `tests/tutorial/test_01_state_reducer.py`
2. `tests/tutorial/test_02_registry_degradation.py`
3. `tests/tutorial/test_03_supervisor_send_fanout.py`
4. `tests/tutorial/test_05_interrupt_resume.py`
5. `tests/tutorial/test_06_reflector_hard_fallback.py`
6. `tests/test_end_to_end_offline.py`

这套顺序的好处是：

- 先懂 reducer，再懂 fan-out/fan-in
- 先懂工具降级，再懂 agent 为什么能稳
- 先懂 interrupt/resume，再懂 HITL 为什么不是“暂停一个 Python 函数”那么简单
- 最后再看完整闭环，认知成本最低

---

## 4. 项目结构怎么分层

目录分层如下：

```text
03_MULTI_AGENT/
├── app/          # 服务入口、生命周期、接口、报告归档、多轮初始化
├── graph/        # LangGraph 状态、路由、主图装配
├── agents/       # Planner / Researcher / Reflector / Writer 节点
├── tools/        # 搜索工具、MCP、KB 适配器、降级链
├── prompts/      # Prompt 模板
├── config/       # 配置与 LLM 工厂
├── scripts/      # CLI 与 smoke test
├── tests/        # 单测 + 教学型测试
└── data/         # 报告输出与 checkpoint 数据
```

最重要的依赖方向是：

```text
app -> graph -> agents -> tools
config 被所有层读取
prompts 被 agents 使用
```

这个依赖方向很值钱，因为它避免了两个常见问题：

- API 层和 agent 逻辑纠缠在一起，最后无法测试
- Tool 直接感知上层流程，导致工具层不可复用

---

## 5. 用一句话理解每一层

### 5.1 `config/`

负责“环境”和“模型工厂”。

- `config/settings.py`：读取 `.env`
- `config/llm.py`：按 `tier=max/turbo` 生成 Qwen 客户端
- `config/tracing.py`：LangSmith tag 预留位

### 5.2 `prompts/`

负责“节点级角色定义”，不负责业务流程。

### 5.3 `graph/`

负责“流程图”和“状态流转规则”。

- `state.py`：定义 `ResearchState` 和 `merge_evidence`
- `router.py`：决定什么时候 fan-out，什么时候写报告
- `workflow.py`：把所有节点装成一张图

### 5.4 `agents/`

负责“每个节点做什么”。

- Planner 产研究计划
- Researcher 拉证据
- Reflector 判断够不够
- Writer 写报告

### 5.5 `tools/`

负责“接外部世界”。

- Tavily / ArXiv / GitHub / DashScope 搜索
- Brave MCP
- 01_RAG 复用
- Internal MCP Server

### 5.6 `app/`

负责“把图做成服务”。

- 启动时注册工具、初始化 checkpointer、编译 graph
- 提供 `/research`、`/resume`、`/turn`
- 写报告到 `data/reports/`

---

## 6. 先抓住核心设计思想

### 6.1 设计思想一：用显式图，而不是让一个 Agent 自己乱跑

这个项目最核心的思想是“流程显式化”。

很多多 Agent Demo 的问题是：

- 看起来 agent 很聪明
- 实际上难以控制
- 失败时不知道卡在哪
- 很难做持久化和恢复

这个项目反过来做：

- 流程节点固定
- 状态字段固定
- 路由规则固定
- 智能只放在必须用 LLM 的地方

所以它更像“有智能节点的工作流系统”，而不是“自由对话的多智能体群聊”。

### 6.2 设计思想二：重推理节点少而精

真正用强模型的地方只有：

- Planner
- Reflector
- Writer

Researcher 节点当前版本默认不做 LLM 二次提炼，而是尽量直接把 tool 结果转成 `Evidence`。这是一种非常现实的降本思路：

- 搜索和抓取已经花了时间
- 再让 LLM 对每一条结果做摘要会放大 token 成本
- 对“研究流水线”来说，先把证据攒够，通常比每步都润色更重要

### 6.3 设计思想三：状态是第一公民

项目不是用“对话历史字符串”维持上下文，而是用 `ResearchState`。

这意味着：

- 哪些字段跨轮保留，哪些字段每轮重置，一清二楚
- 并行节点怎么合并结果，有 reducer 控制
- 中断恢复不是靠前端记忆，而是靠 checkpointer

### 6.4 设计思想四：工具层协议统一，便于降级和替换

`SearchTool` 协议非常关键。因为一旦统一成：

- `name`
- `source_type`
- `search(query, top_k)`
- `close()`

那 HTTP 工具、MCP 工具、RAG 工具都能以同一种方式被 Researcher 消费。

这就是为什么 Web 搜索可以做成一条降级链：

`Tavily -> Brave MCP -> DashScope Search -> 跳过`

### 6.5 设计思想五：把“用户参与”设计进图里

不是让前端自行拦截，而是 Planner 节点直接 `interrupt()`。

这带来的价值是：

- HITL 是流程的一部分，不是外围补丁
- 可以持久化
- 可以恢复
- 可以复盘

---

## 7. 按真实流程讲代码

下面按“用户发请求后，系统内部发生了什么”讲。

### 7.1 第 0 步：进程启动，先把基础设施准备好

入口在 `app/bootstrap.py`。

启动流程：

1. 创建 `ToolRegistry`
2. 注册 Tavily
3. 加载 external MCP
4. 注册 DashScope Search
5. 注册 ArXiv
6. 注册 GitHub
7. 注册 KB Retriever
8. 初始化 `AsyncSqliteSaver`
9. 调 `build_graph()` 编译主图

这里体现出很强的工程意识：

- 工具生命周期统一管理
- HTTP client 和 MCP session 在启动时初始化，避免每次请求重复创建
- checkpointer 是服务级单例，而不是请求级资源

为什么用 `AsyncSqliteSaver` 而不是 `SqliteSaver`：

- 当前图走的是 `graph.ainvoke()`
- 如果 checkpointer 还是同步版本，就会和异步图执行冲突

这是一个典型的真实项目问题，不是教程里那种“写个 demo 就行”。

### 7.2 第 1 步：API 收到请求

入口在 `app/api.py::start_research()`。

它做的事情很简单：

1. 生成 `thread_id`
2. 组装初始 payload
3. 调 `graph.ainvoke(payload, config)`
4. 检查返回里有没有 `__interrupt__`

初始 payload 只有：

```python
{
    "research_query": req.research_query,
    "audience": req.audience,
    "messages": [],
    "evidence": [],
}
```

这说明项目初始状态很克制，没有把一堆默认字段提前塞进去。很多字段都是在节点里按需写入。

### 7.3 第 2 步：主图从 `planner` 开始

主图定义在 `graph/workflow.py`。

主路径是：

```text
START
  -> planner
  -> supervisor
  -> researchers(并行)
  -> reflector
  -> writer
  -> END
```

注意两个关键点：

1. 图入口固定是 `planner`
2. 但后续是否重新规划、是否补查、是否直接收敛，是由路由决定的

### 7.4 第 3 步：Planner 生成研究计划并中断

代码在 `agents/planner.py`。

它做了三件事：

1. 用 `qwen-max` 生成结构化 `ResearchPlan`
2. 调 `interrupt({"phase": "plan_review", "plan": ...})`
3. 用户恢复后，把计划写回 state

这里最重要的不是 prompt，而是结构化输出和 interrupt：

- 结构化输出保证 `plan` 不是脆弱 JSON 字符串
- `interrupt()` 会把当前图停在这个节点
- `Command(resume=...)` 的内容会作为 `interrupt()` 的返回值回流到节点内部

这正是 `tests/tutorial/test_05_interrupt_resume.py` 要验证的核心。

这个设计为什么高级：

- 不是 API 层在“猜” Planner 是否结束
- 而是图自己声明“我现在进入人工确认阶段”

### 7.5 第 4 步：Resume 后，Planner 把用户版本的计划写回 state

`planner_node()` 里用 `_coerce_plan()` 做了一个很实用的工程处理：

- 允许恢复数据是 `ResearchPlan`
- 允许是 dict
- 解析失败则回退原始计划

这类代码的价值在真实项目里非常大，因为前端、API、LangGraph checkpoint 三方之间经常会出现序列化形态差异。

Planner 返回后，state 关键字段变成：

- `plan`
- `plan_confirmed=True`
- `iteration=0`
- `revision_count=0`

### 7.6 第 5 步：Supervisor 节点本身很薄，真正的调度在 Router

这是读这个项目最容易误解的地方。

`agents/supervisor.py` 只是：

- 写 `current_node="supervisor"`
- `iteration + 1`

真正的扇出逻辑在 `graph/router.py::supervisor_route()`。

这个函数根据 state 返回三种结果：

1. `"planner"`
2. `"writer"`
3. `list[Send]`

为什么这是一个很好的设计：

- 节点只负责写状态
- 路由只负责决定走向
- 可测试性更好

你能从 `tests/tutorial/test_03_supervisor_send_fanout.py` 直接看懂这个设计。

### 7.7 第 6 步：`Send` 扇出，进入并行 Researcher

当 `supervisor_route()` 返回 `list[Send]` 时，会把每个子问题按推荐数据源拆成多个任务。

例如：

- `sq1` 推荐 `web + academic`
- `sq2` 推荐 `code`
- `sq3` 推荐 `kb`

最终会生成 4 个 `Send`：

- `web_researcher`
- `academic_researcher`
- `code_researcher`
- `kb_researcher`

每个 `Send` 都带：

- `sub_question`
- `research_query`

这意味着项目的并行粒度不是“按 Agent 一把梭”，而是“子问题 x 数据源”。

这点很关键，因为它决定了系统的可扩展性：

- 后续加新 source_type 很容易
- 后续做 selective re-fanout 也有空间

### 7.8 第 7 步：Researcher 节点只做一件事：取证据

Researcher 的共同逻辑在 `agents/_researcher_base.py`。

核心函数是 `run_research_chain()`。

它的逻辑是：

1. 从 `registry.get_chain(source_type)` 取一条工具链
2. 顺序尝试每个 tool
3. 某个 tool 返回非空结果就短路返回
4. 全部失败则返回空列表

这是一个很标准的“降级链”写法。

它背后的工程思路是：

- 不让某个外部 API 成为单点故障
- 不中断整条研究链路
- 证据宁可少一点，也不要整个系统报错

`tests/tutorial/test_02_registry_degradation.py` 正是在保护这个约束。

### 7.9 第 8 步：Researcher 为什么要加 `safe_node`

四个 Researcher 都套了：

- `@with_tags(...)`
- `@safe_node`

其中 `safe_node` 非常重要。

它把节点异常变成：

```python
{
    "evidence": [],
    "messages": [AIMessage(content="[skip] ...")]
}
```

这意味着：

- 某个工具坏了，主图不死
- Reflector 还能继续评估“证据不够”
- 系统行为从“硬失败”变成“软降级”

这类模式在生产里比“全部异常直接 raise”实用得多。

### 7.10 第 9 步：证据怎么汇总，为什么不是 `operator.add`

关键代码在 `graph/state.py::merge_evidence()`。

这一步体现了 LangGraph 的真正价值：并行节点不是自己写 merge 逻辑，而是通过 reducer 自动聚合。

这里没有使用简单的列表拼接，而是做了三件事：

1. 按 `source_url` 去重
2. 同 URL 保留更高 `relevance_score`
3. 最终按得分倒序

为什么这比 `operator.add` 高级：

- 并行搜索时经常多个工具命中同一来源
- 如果不去重，Writer 会拿到重复证据
- 重复证据会放大噪音、浪费 token、削弱引用质量

这就是 `tests/tutorial/test_01_state_reducer.py` 反复验证的地方。

### 7.11 第 10 步：Reflector 负责判断“够不够”

代码在 `agents/reflector.py`。

Reflector 不是简单地“再总结一次”，它承担两个系统级职责：

1. 质量控制
2. 成本控制

它先把 `plan` 和 `evidence` 压缩成摘要，再让强模型输出 `ReflectionResult`：

- `coverage_by_subq`
- `missing_aspects`
- `next_action`
- `additional_queries`

然后把结果写回 state。

这里真正要学的是“反思节点”的作用：

- 不是为了显得多 Agent 很高级
- 而是为了把“补查”做成一个显式可控回路

### 7.12 第 11 步：为什么第 3 轮强制收敛

`reflector.py` 里有：

```python
if rc >= MAX_REVISION:
    return {"next_action": "force_complete", ...}
```

这一步非常像真实业务系统里的兜底逻辑。

原因很直接：

- 如果没有上限，反思可能一直觉得“不够”
- 多轮补查会迅速放大耗时和成本
- 用户需要的是“足够好并按时返回”，不是“无限追求完美”

这也是 `tests/tutorial/test_06_reflector_hard_fallback.py` 专门保护的逻辑。

如果面试官问你“为什么不一直循环直到满意”，标准回答就是：

- 研究任务存在收益递减
- 系统必须有 SLA 和成本边界
- 所以要显式设迭代上限

### 7.13 第 12 步：Writer 不是从 0 编造，而是拿编号证据写报告

代码在 `agents/writer.py`。

它先做一件很重要的事情：

- 给所有 `Evidence` 编号
- 生成 `numbered_evidence`
- 让模型按 `[^N]` 引用

同时后端直接把 citation 列表从 evidence 生成出来，而不是全信模型。

这是一种典型的“模型写正文，系统掌握结构事实”的设计。

优点：

- 引用编号和 URL 可以一一对应
- 模型不需要自己创造引用结构
- 更容易做后处理和归档

如果模型漏了引用章节，代码还会自动补 `## 引用`。

这说明 Writer 不是“纯 prompt 工程”，而是“LLM + deterministic post-processing”。

### 7.14 第 13 步：报告怎么落盘

落盘逻辑在 `app/report_store.py`。

文件名格式：

```text
{YYYYMMDD-HHMMSS}_{slug(query)}_{thread_id}.md
```

这么设计的好处：

- 人能看懂
- 可以按 thread_id 反查
- 可以按时间排序
- 不依赖数据库也能快速浏览历史报告

这是一个很实用的作品集设计，面试展示非常友好。

### 7.15 第 14 步：多轮追问怎么做

入口在 `app/api.py::turn_research()`。

它没有新建会话，而是：

1. 复用原 `thread_id`
2. 调 `reset_per_turn()`
3. 设置 `plan_confirmed=False`
4. 再次进入图

`app/turn_init.py::reset_per_turn()` 的思路很值得学。

保留：

- `messages`
- `evidence`
- `plan`
- `final_report`

重置：

- `revision_count`
- `iteration`
- `coverage_by_subq`
- `missing_aspects`
- `next_action`

这说明项目对“会话记忆”和“轮次过程变量”做了明确切分。

---

## 8. `ResearchState` 为什么是项目中最重要的文件之一

文件：`graph/state.py`

很多人读多 Agent 项目，只盯着 prompt 和 graph。其实这个项目更值得看的，是 `ResearchState`。

它定义了系统真正共享的业务事实。

### 8.1 关键字段按职责分组

输入相关：

- `research_query`
- `audience`

计划阶段：

- `plan`
- `plan_confirmed`

调研阶段：

- `evidence`
- `revision_count`

反思阶段：

- `coverage_by_subq`
- `missing_aspects`
- `next_action`
- `additional_queries`

输出阶段：

- `final_report`
- `citations`
- `report_path`

调度与对话：

- `messages`
- `current_node`
- `iteration`

### 8.2 为什么 `evidence` 要用 reducer

因为并行节点会同时写这个字段。

没有 reducer 的情况下，你要么：

- 手写 fan-in 节点自己 merge

要么：

- 后写入的节点覆盖前面的值

两种都很差。

LangGraph 的 reducer 让“并行结果聚合”变成状态层能力，而不是业务节点临时补丁。

---

## 9. Tool 层怎么设计，为什么很像真实工作里的基础设施层

### 9.1 统一协议是这层的灵魂

文件：`tools/base.py`

`SearchTool` 协议统一了所有检索型工具。

这意味着上层 Researcher 并不关心：

- 你是 HTTP API
- 你是 MCP server
- 你是本地 RAG

它只关心：给我结果。

### 9.2 Registry 的价值不在“存工具”，而在“定义顺序”

文件：`tools/registry.py`

`ToolRegistry` 很简单，但很有用。

因为它不仅保存工具，更表达了一条业务策略：

- 哪个是主工具
- 哪个是 fallback
- 哪个最后兜底

顺序本身就是策略。

### 9.3 HTTP 工具实现为什么值得看

文件：

- `tools/_http.py`
- `tools/tavily_tool.py`
- `tools/arxiv_tool.py`
- `tools/github_tool.py`
- `tools/dashscope_search_tool.py`

这些实现里有几个典型工程细节：

- 共用 `httpx.AsyncClient`
- tenacity 做指数退避
- timeout 不同工具可单独调整
- snippet 长度截断，避免污染后续 prompt
- relevance_score 统一成浮点数，便于排序

这说明工具层不是“能返回就行”，而是在为后面的 State 和 Writer 做规范化。

### 9.4 为什么 DashScope Search 是很现实的兜底

设计上它有两个价值：

1. 国内网络环境更稳
2. 不依赖额外搜索服务就能兜底

项目里还刻意区分了两个 DashScope 用法：

- LLM 走 OpenAI compatible endpoint
- 内置搜索走 DashScope 原生 endpoint

这不是绕，而是因为原生端点才能返回 `search_info.search_results`。

这类“同一个供应商不同端点承担不同职责”的做法，在真实项目里非常常见。

### 9.5 `kb_retriever.py` 为什么是本项目最有工程味的文件之一

这是一个非常值得在面试中讲的点。

项目没有直接复制 `01_RAG` 代码，也没有强行把 `01_RAG` 加进全局 `PYTHONPATH`，而是用了一个非常克制的适配策略：

- 临时调整 `sys.path`
- 暂时移走冲突模块
- 导入 `rag.retriever`
- 再恢复现场

它解决的是“复用历史项目，但不污染当前项目依赖空间”的问题。

这和真实工作里“老系统模块复用”“单仓多子项目共存”非常像。

如果你能把这个文件讲清楚，面试官会认为你不仅会写 agent，还能处理复杂 Python 工程边界。

### 9.6 External MCP 和 Internal MCP 的价值

External MCP：

- 项目作为客户端，去消费别人的能力
- 当前主要是 Brave Search

Internal MCP：

- 项目作为服务端，把自己的能力暴露出去
- 提供 `kb_search`、`list_reports`、`read_report`、`list_evidence`、`trigger_research`

这意味着项目不是封闭应用，而是“可被其他 agent 系统调用的能力节点”。

这是很强的架构升级点。

---

## 10. 为什么这个项目适合拿来讲 LangGraph

### 10.1 它覆盖了 LangGraph 最关键的几个能力

- `StateGraph`
- `TypedDict state`
- reducer
- `Send`
- `interrupt()`
- `Command(resume=...)`
- checkpointer
- 多轮恢复

### 10.2 它不是把 LangGraph 当成“另一种 Agent SDK”

很多项目只是把 LangGraph 当作“能连几个节点的框架”。这个项目更进一步：

- 把状态视为核心
- 把并行与 reducer 结合
- 把 HITL 放进图里
- 把多轮恢复建立在 checkpointer 上

所以它更像“工作流引擎 + 智能节点”的用法。

---

## 11. 技术栈为什么这么选

这一节是你在实际工作里最该会讲的。

### 11.1 为什么选 LangGraph，不选纯 AutoGen/CrewAI 风格

因为这个场景最关键的是：

- 流程可控
- 状态可恢复
- 并行可聚合
- HITL 可插入
- 多轮会话可持久化

如果你只是做“多 agent 对话演示”，AutoGen/CrewAI 很直观；但如果你做的是生产研究流水线，LangGraph 的状态图优势更明显。

实际工作里的选型话术可以这么说：

> 当流程存在明确阶段、需要中断恢复、需要持久化和回放时，我优先选显式图编排，而不是自由对话式 agent 框架。

### 11.2 为什么选 FastAPI

因为它适合做：

- 异步接口
- SSE
- Pydantic schema
- 生命周期管理

如果后续要接 Web UI、做 streaming、挂更多管理接口，FastAPI 成本最低。

### 11.3 为什么选 SQLite checkpointer

当前阶段目标是：

- 本地可跑
- 作品集易部署
- 会话持久化够用

SQLite 足够。

真实工作里如果要上多实例部署，通常会升级为：

- Postgres
- Redis + durable store
- 或 LangGraph 支持的集中式持久化方案

### 11.4 为什么选 Qwen + DashScope

这是一个非常现实的中国团队选型：

- 国内可达性好
- 中文能力强
- 成本可控
- 兼容 OpenAI 协议，能接 LangChain

同时项目把 LLM 调用和搜索兜底拆开，说明团队不是“迷信一个平台包打天下”，而是按能力分工使用。

### 11.5 为什么选官方 MCP SDK

因为它在标准化生态里有长期价值：

- 统一工具协议
- 便于接 Claude Desktop / Cursor / 其他客户端
- 未来能力能复用，不被当前 App 形态绑死

---

## 12. 结合实际工作讲技术选型与调优

这一节是你要求里的重点。

### 12.1 模型调优：把强模型用在刀刃上

当前实现的思路是：

- Planner：`qwen-max`
- Reflector：`qwen-max`
- Writer：`qwen-max`
- 其他环节尽量不用强模型

这非常符合真实工作：

- 规划、判断覆盖度、长文写作，最吃推理能力
- 搜索结果转证据，没必要每次都用大模型做润色

如果线上成本过高，优先调这几项：

1. 降低 Writer 输入证据条数
2. 限制 `snippet` 长度
3. Planner 从 5 个子问题压到 3 个
4. Reflector 只在 evidence 数量较少时启用强模型
5. 对简单 query 走 quick depth

### 12.2 搜索链调优：用“多源降级”代替“单源高可用幻想”

真实工作里，外部搜索服务经常会遇到：

- 配额问题
- 区域网络问题
- API 波动

这个项目没有强求 Brave 一定成功，而是做成：

`Tavily -> Brave MCP -> DashScope Search`

这就是务实调优。

如果是企业生产环境，我会继续加：

- 每个 provider 的成功率监控
- 按错误类型切换降级路径
- 针对 429 做更细粒度 backoff
- 针对 query 类型做 provider routing

### 12.3 并发调优：先把 fan-out 做出来，再加限流

当前项目已经有并行扇出，但还没做统一 semaphore。

这在 Demo 阶段是合理的，因为先证明能力，后做细调。

如果进入线上，我会加：

- 全局并发上限
- 单 provider 并发上限
- 热点 query 缓存
- 相同 source_url 的去重缓存

为什么？

- LLM 不是唯一的瓶颈，外部 API 配额常常才是
- 没有限流时，并行会放大失败率

### 12.4 状态调优：为什么 reducer 要去重

在真实工作里，证据质量不只是“多”，而是“有效多样”。

如果不按 URL 去重，会出现：

- Writer 输入长度膨胀
- 相同观点被重复当作多条证据
- 反思节点误以为覆盖度很高

所以 `merge_evidence()` 其实是质量调优，不只是代码美观。

### 12.5 持久化调优：为什么多轮必须靠 checkpoint，不靠前端存历史

如果只靠前端把上一轮报告再传回来，会有几个问题：

- 报告不是完整状态
- evidence 结构会丢
- 用户刷新页面可能丢会话
- interrupt/resume 无法自然恢复

真实工作里的原则是：

> 前端保存展示态，后端保存执行态。

这个项目就是这么做的。

### 12.6 复用调优：为什么不复制 01_RAG

很多团队为了快，会直接复制旧项目的 RAG 代码。

短期看省事，长期看问题很大：

- bug 修两份
- 依赖漂移
- 行为不一致

这个项目选择写适配器，这就是更成熟的技术债控制方式。

### 12.7 Prompt 与后处理调优：不要把所有可靠性都押给模型

Writer 的实现很说明问题：

- 编号 evidence
- 让模型引用编号
- 后端自己生成 citation
- 缺引用章节就自动补

这才是生产式 prompt 工程。

真实工作里如果只靠 prompt 让模型“请务必正确引用”，最后通常会踩坑。

---

## 13. 这个项目最值得你在面试里强调的亮点

### 13.1 亮点一：它不是玩具多 Agent，而是可恢复、可中断、可扩展的研究流水线

关键词：

- HITL
- 持久化
- 并行 fan-out
- reducer 聚合
- 反思补查

### 13.2 亮点二：把工具层做成了协议和降级链

这能证明你理解：

- 抽象边界
- 稳定性设计
- provider fallback

### 13.3 亮点三：复用了旧项目能力而不是重复造轮子

`01_RAG` 复用是很好的“工程复用能力”证明。

### 13.4 亮点四：支持双向 MCP

这能证明你不只会“调工具”，还会“把自己的系统变成工具”。

### 13.5 亮点五：测试不是摆设

当前有 35 个测试，且很多是教学型测试，说明作者不是只追求跑通，而是明确在保护核心机制。

---

## 14. 你可以怎么讲“从代码看流程”

面试时建议用下面这个版本：

> 用户先通过 `/research` 发起研究请求，系统创建 `thread_id` 并进入 LangGraph 主图。`planner_node` 用强模型把问题拆成结构化 `ResearchPlan`，随后在图内部调用 `interrupt()` 进入人工确认。用户修改计划后，后端通过 `Command(resume=...)` 恢复执行。  
> 恢复后 `supervisor_route()` 根据 `plan` 把子问题按推荐数据源展开成多个 `Send` 任务，4 个 Researcher 并行取证。Researcher 本身不感知具体 provider，只通过 `ToolRegistry` 走降级链。并行结果通过 `merge_evidence()` 自动聚合、去重、排序。  
> 接着 `reflector_node` 判断证据覆盖是否充分，如果不足就触发下一轮补查，但最多 3 轮，防止成本失控。最后 `writer_node` 基于编号证据生成 Markdown 报告，并由后端补齐引用与归档。后续追问复用同一个 `thread_id`，通过 checkpoint 恢复历史 evidence 和计划，实现多轮深挖。

如果你能把这段逻辑顺着说完，基本已经超过很多只会展示架构图的人。

---

## 15. 面试高频问题与回答

### Q1：为什么要用多 Agent，而不是一个大模型一次写完报告？

答：

因为深度研究任务天然分阶段：

- 先拆问题
- 再分源取证
- 再判断覆盖度
- 最后写报告

如果一个模型一次完成，常见问题是：

- 覆盖不全
- 没有显式证据收集过程
- 用户无法中途纠偏
- 难以多轮复用中间状态

多 Agent 在这里不是为了炫技，而是为了把复杂任务分成可控步骤。

### Q2：为什么 Supervisor 这么薄？

答：

这是刻意设计。当前版本把“智能调度”尽量下沉到显式路由规则里，让系统更可预测、更可测。Supervisor 节点只负责状态推进，实际扇出由 `supervisor_route()` 决定。这样比“让一个 LLM 每轮自由决定下一步”更稳定。

### Q3：为什么 Researcher 不直接用 LLM 总结搜索结果？

答：

这是成本和收益权衡。当前版本优先保证证据收集闭环和多源覆盖，因此默认直接把工具结果结构化成 `Evidence`。如果每个搜索结果都再做 LLM 摘要，token 成本会明显放大，而且不一定显著提升最终报告质量。

### Q4：为什么 `merge_evidence()` 不用简单拼接？

答：

因为多源检索会命中重复 URL。简单拼接会导致：

- Writer 输入膨胀
- 重复证据影响覆盖度判断
- 引用质量下降

所以 reducer 按 URL 去重、保留更高分结果，是质量控制的一部分。

### Q5：为什么要有 Reflector？

答：

Reflector 的作用不是“再总结一次”，而是显式判断证据覆盖是否足够，并决定要不要补查。它把“研究质量控制”从隐性 prompt 变成了显式节点。

### Q6：为什么要限制反思轮数？

答：

为了控制成本和延迟。真实研究任务存在收益递减，系统需要在“充分性”和“SLA”之间做权衡。所以设置 `max_revision=3`，第 3 轮强制收敛。

### Q7：为什么要用 `interrupt()`，直接 API 返回 plan 不行吗？

答：

如果把计划确认放在图外，流程状态就会断裂。`interrupt()` 的好处是：

- 中断点由图自己定义
- checkpoint 能保存执行态
- `resume` 能自然回到同一个节点继续

这是 LangGraph 很有价值的能力。

### Q8：为什么选择 LangGraph？

答：

因为这个项目需要：

- 显式状态管理
- 条件路由
- 并行 fan-out/fan-in
- HITL 中断恢复
- 持久化 checkpoint

这些是 LangGraph 的强项。

### Q9：为什么选 SQLite 做持久化？

答：

当前目标是本地可跑和作品集展示，SQLite 足够轻量。它能满足单机、多轮恢复、interrupt/resume 的需求。如果进入多实例生产，再切更强的持久化后端。

### Q10：为什么要做 ToolRegistry？

答：

它把 provider 选择从业务逻辑里抽出来。Researcher 只认 `source_type`，不认具体厂商。这样切换主工具、加 fallback、做 A/B 测试都更容易。

### Q11：为什么既做 external MCP，又做 internal MCP？

答：

因为系统既要消费外部能力，也要把自身能力输出给外部 agent 生态。前者体现兼容性，后者体现复用价值。

### Q12：`01_RAG` 为什么不直接复制过来？

答：

复制会造成长期维护问题。当前实现通过适配器和隔离 import 机制复用旧项目能力，避免了代码分叉，也降低了依赖污染风险。

### Q13：这个项目最像真实工作的部分是什么？

答：

不是 prompt，而是这些地方：

- provider fallback
- checkpoint 恢复
- interrupt/resume
- 旧系统能力适配
- 节点级容错
- 成本与质量之间的权衡

### Q14：如果让你继续做下一步，你会做什么？

答：

优先做三件事：

1. SSE 流式事件，让前端能看到节点进度
2. selective re-fanout，只补查缺失子问题
3. 评测体系和 tracing，把质量和成本数据化

### Q15：这个项目有哪些当前不足？

答：

- Supervisor 还没有真正的动态智能路由
- SSE 和 UI 尚未完成
- selective re-fanout 还没做
- LangSmith 还没真正接通
- 线程列表和会话管理还偏轻量

能主动讲不足，往往比只讲优点更像真实工程师。

---

## 16. 读代码时你最该重点看的文件

如果只给你 10 个文件，我建议看这 10 个：

1. `graph/state.py`
2. `graph/router.py`
3. `graph/workflow.py`
4. `agents/planner.py`
5. `agents/_researcher_base.py`
6. `agents/reflector.py`
7. `agents/writer.py`
8. `tools/registry.py`
9. `tools/kb_retriever.py`
10. `app/api.py`

为什么是这 10 个：

- 前 3 个决定流程
- 中间 4 个决定业务价值
- 后 3 个决定工程质量

---

## 17. 最适合你自己复盘的方式

建议你按下面的方法自己再走一遍：

1. 先跑 `pytest 03_MULTI_AGENT/tests -q`
2. 再读 tutorial tests
3. 再用 `python -m scripts.run_local "你的问题"` 跑一遍真实链路
4. 然后手动画出 state 在每一步写了哪些字段
5. 最后自己口述一遍完整流程

你真正掌握这个项目的标准不是“能看懂代码”，而是：

- 你能解释为什么要这样拆
- 你能说清每个节点写什么状态
- 你能说明哪些设计是在控制成本、哪些设计是在提升稳定性

---

## 18. 最后给你的项目总结模板

如果你要把这个项目写进简历或面试陈述，可以直接参考这版：

> 我做了一个基于 LangGraph 的多 Agent 深度研究系统，支持研究计划拆解、人工确认、并行多源调研、反思补查、报告生成和多轮追问。系统采用显式 `ResearchState` 管理流程状态，用 `interrupt()/Command(resume)` 实现 HITL，用 `Send + reducer` 实现并行 fan-out/fan-in，用 SQLite checkpointer 支持跨轮恢复，并通过统一 `SearchTool` 协议把 Tavily、GitHub、ArXiv、DashScope Search、MCP 和本地 RAG 接到一套降级链里。工程上还做了节点级容错、报告归档、旧项目 RAG 复用和 internal/external MCP 双向集成。

这段话的重点是：

- 说能力
- 说机制
- 说工程化

不要只说“我做了 7 个 agent”，那样信息量太低。

---

## 19. 附：当前项目真实运行与测试信息

我基于当前仓库核对到的事实：

- 当前已有样例报告落在 `data/reports/`
- 当前测试结果为 `35 passed`
- API 已实现：`/research`、`/resume`、`/turn`、`/state`、`/threads`、`/reports`
- SSE `/stream` 还未落地

因此，学习和面试时要以“当前实现”作为主线，以“设计文档中的下一步规划”作为扩展项，不要把未实现功能当成已交付功能去讲。

---

## 20. 一句话收尾

这个项目最值得学的，不是“怎么堆 7 个 Agent”，而是“如何把一个复杂研究任务拆成一条可中断、可恢复、可并行、可降级、可复用的工程化流水线”。
