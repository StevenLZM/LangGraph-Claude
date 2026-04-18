# PROJECT SCENARIO：InsightLoop — AI 行业深度研究多 Agent 系统

> 项目编号：03 | 难度：⭐⭐⭐⭐⭐ | 预计周期：3-4 周
> 类型：**生产级多轮对话 Multi-Agent 系统**
> 文档版本：v1.0  ·  创建日期：2026-04-18

---

## 一、一句话定位

**InsightLoop** 是一个面向 AI 从业者的**多轮对话式深度研究助手**。用户用自然语言提出研究问题（如"分析 2025 年开源 Agent 框架格局"），系统由 7 个专业 agent 协作完成"**计划 → 人工确认 → 并行调研 → 反思补查 → 报告生成 → 多轮追问**"全流程，本地 Docker 一键启动，可作为 AI Agent 工程师面试的核心亮点项目。

---

## 二、项目背景与目标

### 2.1 真实痛点

人工做一次行业深度研究通常要：搜资料 → 读论文 → 翻 GitHub → 整合写报告，单次耗时 4-8 小时。市面已有的 ChatGPT/Claude 单 agent 模式存在三个痛点：

1. **覆盖不全**：单 agent 无法同时兼顾"网搜+学术+代码+私有知识库"多源信息
2. **跑偏浪费**：用户没机会确认研究方向，跑完才发现方向错了，token/时间双浪费
3. **无法多轮深挖**：一次性输出报告后无法基于前文上下文继续追问

### 2.2 目标

构建一个**多 agent 协作 + 多轮对话 + 生产级可运行**的深度研究系统，做到：

- 系统自动拆解研究问题，并行调用多源 agent 收集证据
- 用户在关键节点（计划确认）参与决策，避免方向跑偏
- 报告产出后支持基于完整上下文的多轮追问
- 本地一键启动，对外提供 SSE 流式 API + Web UI

### 2.3 学习目标（对标 AI Agent 工程师面试技术栈）

| 技术栈维度 | 在本项目中的覆盖 |
|---|---|
| **LangGraph 核心** | StateGraph、嵌套子图、条件路由、Send 并行、循环控制 |
| **Multi-Agent 编排** | Supervisor 中心调度模式 + 7 个 sub-agent 协作 |
| **Human-in-the-Loop** | `interrupt()` + `Command(resume=...)` 在计划阶段做人工确认 |
| **状态管理与持久化** | TypedDict State、Reducer、SqliteSaver checkpointer、thread_id 隔离会话 |
| **多轮对话** | 跨 turn 状态恢复，复用历史 evidence 做追问 |
| **结构化输出** | `with_structured_output(Pydantic)` 替代 JSON 解析 |
| **Tool Use & MCP** | Tavily / ArXiv / GitHub API / 本地 RAG / MCP servers (Brave、Filesystem) |
| **RAG 集成** | 复用 01_RAG 的混合检索 (向量 + BM25 + RRF) 作为知识库 sub-agent |
| **流式输出** | `astream_events` + FastAPI SSE 实时推送 agent 进度 |
| **可观测性** | LangSmith tracing，给每个 agent 打 tag |
| **生产化** | FastAPI + Pydantic v2 + SQLite + Docker Compose 一键启动 |
| **评测** | 内置评测集 + LLM-as-judge，量化覆盖度/准确性/引用质量 |

---

## 三、核心交互流程（多轮对话）

```
┌──────────────────────────────────────────────────────────────────┐
│ Turn 1                                                            │
│                                                                   │
│ 用户："帮我分析 2025 年开源 Agent 框架格局，                     │
│        重点对比 LangGraph / AutoGen / CrewAI"                     │
│                              │                                    │
│                              ▼                                    │
│           ┌──────────────────────────────────────┐                │
│           │ [Planner Agent] 拆解研究计划：       │                │
│           │  ① 各框架 GitHub stars/contributor    │                │
│           │  ② 核心抽象差异（state/handoff/role） │                │
│           │  ③ 学术界引用情况                    │                │
│           │  ④ 生产案例与社区生态                │                │
│           └──────────────────────────────────────┘                │
│                              │                                    │
│                  ⏸  INTERRUPT  ⏸                                  │
│                              │                                    │
│ 用户："去掉③，加一个'国内厂商动态'"                              │
│                              │                                    │
│                              ▼                                    │
│           ┌──────────────────────────────────────┐                │
│           │ [Supervisor] 并行派发 sub-agents：    │                │
│           │  ├─ Web Researcher       → ①, ④      │                │
│           │  ├─ Code Researcher      → ①, ②      │                │
│           │  ├─ KB Researcher (RAG)  → ②         │                │
│           │  └─ Web Researcher (新闻)→ 国内厂商  │                │
│           └──────────────────────────────────────┘                │
│                              │ evidence 汇总                      │
│                              ▼                                    │
│           ┌──────────────────────────────────────┐                │
│           │ [Reflector] 审查覆盖度：              │                │
│           │  发现"缺少各框架性能基准对比"         │                │
│           │  → 触发补查（回到 Supervisor）        │                │
│           └──────────────────────────────────────┘                │
│                              │ (max_iterations=3 兜底)            │
│                              ▼                                    │
│           ┌──────────────────────────────────────┐                │
│           │ [Report Writer] 汇总成结构化报告      │                │
│           │  Markdown + 引用脚注 + 元数据         │                │
│           └──────────────────────────────────────┘                │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────┐
│ Turn 2                                                            │
│ 用户："再深入对比 LangGraph 和 AutoGen 的 state 管理"            │
│                                                                   │
│ → 复用 thread_id 加载历史 state（含已收集的 evidence）            │
│ → Supervisor 决定是否需要补充调研，或直接基于已有证据深挖         │
│ → Report Writer 输出针对性补充分析                                │
└───────────────────────────────────────────────────────────────────┘
```

---

## 四、Agent 分工设计（7 个）

### 4.1 Supervisor（中心调度）

- **职责**：每一轮决定下一步执行哪个 sub-agent / 是否进入反思 / 是否结束
- **实现**：LangGraph `add_conditional_edges` + LLM 路由（带 fallback 到规则路由）
- **输入**：当前 ResearchState
- **输出**：`next_node: Literal["planner","web","academic","code","kb","reflector","writer","end"]`

### 4.2 Planner

- **职责**：把用户研究问题拆解成 3-6 个可执行子问题，每个子问题推荐数据源
- **结构化输出**：
  ```python
  class ResearchPlan(BaseModel):
      sub_questions: list[SubQuestion]
      estimated_depth: Literal["quick", "standard", "deep"]
  class SubQuestion(BaseModel):
      id: str
      question: str
      recommended_sources: list[Literal["web","academic","code","kb"]]
  ```
- **关键点**：输出后**立即触发 `interrupt()`** 等待用户确认/修改

### 4.3 Web Researcher

- **职责**：实时网搜，针对子问题提取要点 + 引用
- **工具**：Tavily（主）+ Brave Search via MCP（兜底）
- **输出**：`list[Evidence]`，每条 evidence 包含 source_url、snippet、relevance_score

### 4.4 Academic Researcher

- **职责**：学术论文检索（ArXiv），提取摘要 + 引用
- **工具**：ArXiv API
- **输出**：同 Web Researcher 格式，source_type="academic"

### 4.5 Code Researcher

- **职责**：调研代码仓库元数据 + 关键代码片段
- **工具**：GitHub API（stars/issues/PR/release 趋势）+ 文件抓取
- **输出**：包含 repo metadata 和 code snippet 的 evidence

### 4.6 KB Researcher（复用 01_RAG）

- **职责**：本地知识库混合检索（白皮书、技术手册、内部文档）
- **工具**：**复用 01 的混合检索**（Chroma 向量 + BM25 + RRF 融合）
- **价值**：体现项目间技术复用，且能让用户上传自己的资料

### 4.7 Reflector

- **职责**：审查已收集 evidence 对所有子问题的覆盖度，输出补查指令
- **结构化输出**：
  ```python
  class ReflectionResult(BaseModel):
      coverage_by_subq: dict[str, int]  # 0-100
      missing_aspects: list[str]
      next_action: Literal["sufficient", "need_more_research", "force_complete"]
      additional_queries: list[str] | None
  ```
- **兜底**：`revision_count >= 3` 强制走到 writer

### 4.8 Report Writer

- **职责**：汇总所有 evidence，生成结构化 Markdown 报告
- **要求**：标题层级清晰、关键论点配引用脚注 `[^1]`、末尾附完整引用列表
- **输出**：`final_report: str` + `citations: list[Citation]`

---

## 五、LangGraph 技术展示点（面试核心）

| 技术点 | 在本项目中的体现 |
|---|---|
| **StateGraph + 嵌套子图** | 主图 = Supervisor 调度；4 个 Researcher 打包成可并行 sub-graph |
| **条件路由** | Supervisor 输出 `next_node` 决定走向；Reflector 输出 `next_action` 决定补查/结束 |
| **循环控制** | Researcher → Reflector → Supervisor 形成补查循环，max_iterations 兜底防止死循环 |
| **并行 fan-out / fan-in** | 用 `Send` API 把多个子问题并行派发给不同 Researcher，结果用 reducer 聚合 |
| **interrupt + Command(resume)** | 计划阶段触发 `interrupt({"plan": ...})`，前端展示后用 `Command(resume={"plan": modified})` 恢复 |
| **checkpointer** | `SqliteSaver` 持久化对话 state；同 thread_id 跨 turn 恢复历史 |
| **Streaming** | `astream_events` 监听 node 进入/退出/工具调用，转 SSE 推前端 |
| **结构化输出** | 所有 agent 用 `llm.with_structured_output(PydanticModel)`，告别脆弱的 JSON 解析 |
| **错误处理** | 节点级 try/except + 最大重试 + 降级策略（如某 sub-agent 失败时跳过但记录） |

---

## 六、核心状态设计

```python
# graph/state.py
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class Evidence(BaseModel):
    sub_question_id: str
    source_type: Literal["web", "academic", "code", "kb"]
    source_url: str
    snippet: str
    relevance_score: float
    fetched_at: str

class SubQuestion(BaseModel):
    id: str
    question: str
    recommended_sources: list[str]
    status: Literal["pending", "researching", "done"]

class ResearchState(TypedDict):
    # === 输入 ===
    research_query: str
    audience: str

    # === 计划阶段 ===
    plan: list[SubQuestion] | None
    plan_confirmed: bool

    # === 调研阶段（evidence 用 reducer 累加） ===
    evidence: Annotated[list[Evidence], add]
    revision_count: int

    # === 反思阶段 ===
    coverage_by_subq: dict[str, int]
    missing_aspects: list[str]

    # === 输出阶段 ===
    final_report: str
    citations: list[dict]

    # === 多轮对话 ===
    messages: Annotated[list[BaseMessage], add_messages]

    # === 调度 / 追踪 ===
    next_node: str
    current_node: str
    iteration: int
```

---

## 七、技术栈

| 层次 | 选型 | 备注 |
|---|---|---|
| 编排 | **LangGraph 0.2+** | StateGraph、SqliteSaver、Send、interrupt |
| LLM | **DashScope（阿里云百炼）** | 通过 `langchain-community` 的 `ChatTongyi` 或 OpenAI 兼容接口调用；主用 qwen-max 等高能力模型，路由/反思等轻量节点用 qwen-turbo 降本；API Key 后续通过 `.env` 注入 |
| 网搜 | Tavily API | 主搜索源 |
| 学术 | ArXiv API | 论文检索 |
| 代码 | GitHub REST API | 仓库元数据 + 文件抓取 |
| 私有 RAG | **复用 01_RAG**（Chroma + BM25 + RRF） | 体现项目间技术延续 |
| MCP | Brave Search、Filesystem | 兜底搜索 + 报告归档 |
| 后端 | FastAPI + Pydantic v2 | SSE 流式推送 |
| 前端 | Streamlit | 聊天界面 + 实时 agent 状态侧栏 + 报告预览 |
| 持久化 | SQLite (checkpointer) + 文件系统 (报告 + 引用快照) | 中等生产化定位，避免与 05 的 Postgres 重复 |
| 可观测性 | LangSmith tracing | 给每个 agent 打 tag，可视化执行路径 |
| 评测 | 自建 20 题评测集 + LLM-as-judge | 覆盖度/准确性/引用质量三维度评分 |
| 部署 | Docker Compose 一键启动 | app + chroma 两个 service |

---

## 八、目录结构

```
03_MULTI_AGENT/
├── app/
│   ├── api.py                  # FastAPI + SSE
│   ├── streamlit_ui.py         # 前端
│   └── schemas.py              # API Pydantic models
├── graph/
│   ├── state.py                # ResearchState
│   ├── workflow.py             # 主图构建
│   ├── router.py               # Supervisor 路由
│   └── nodes_parallel.py       # Send fan-out/in 逻辑
├── agents/
│   ├── supervisor.py
│   ├── planner.py
│   ├── researcher_web.py
│   ├── researcher_academic.py
│   ├── researcher_code.py
│   ├── researcher_kb.py
│   ├── reflector.py
│   └── writer.py
├── tools/
│   ├── tavily_tool.py
│   ├── arxiv_tool.py
│   ├── github_tool.py
│   ├── kb_retriever.py         # 复用 01_RAG 的混合检索
│   └── mcp_loader.py
├── rag/                        # 软链/复用 01_RAG 模块
├── prompts/
│   └── templates.py
├── config/
│   ├── llm.py                  # 模型配置
│   ├── tracing.py              # LangSmith
│   └── settings.py
├── evals/
│   ├── dataset.jsonl           # 20 题评测集
│   ├── judge.py                # LLM-as-judge
│   └── run_eval.py
├── data/
│   ├── documents/              # 私有知识库 PDF
│   └── reports/                # 生成的报告归档
├── tests/
├── .env.example
├── .mcp.json
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 九、功能需求清单

### 9.1 核心功能

- **F01** 用户在 Web UI 输入研究问题，可选择研究深度（quick / standard / deep）
- **F02** 系统生成研究计划后**暂停**，前端展示计划，用户可编辑/确认/中止
- **F03** 用户确认后，系统并行派发 sub-agent，前端实时展示每个 agent 状态
- **F04** Reflector 审查后自动决定是否补查，最多 3 轮，超过则强制出报告
- **F05** 最终报告以 Markdown 渲染，含引用脚注，可导出 .md
- **F06** 同一会话内支持多轮追问，复用历史 evidence
- **F07** 报告自动归档到 `data/reports/`，文件名含时间戳和主题

### 9.2 生产级要求

- **F08** SSE 流式推送 agent 进入/退出/工具调用事件
- **F09** SQLite checkpointer 持久化会话，浏览器刷新后可继续
- **F10** thread_id 隔离不同会话；支持列出历史会话
- **F11** 节点级错误捕获 + 单 sub-agent 失败时跳过并记录
- **F12** Docker Compose 一键启动，附 `.env.example`
- **F13** LangSmith 追踪每次执行，可在 UI 中点击查看 trace 链接

### 9.3 评测

- **F14** 提供 20 题评测集（覆盖技术、产业、对比、追问四类）
- **F15** LLM-as-judge 三维度打分（覆盖度/准确性/引用质量）
- **F16** `make eval` 一键跑完整评测集，输出 markdown 报告

---

## 十、评估标准（项目验收）

- [ ] 完整跑通 7 agent 协作流程，最终报告 ≥ 1500 字、≥ 5 条引用
- [ ] 计划确认 interrupt 在前端正常触发，用户编辑后能正确恢复
- [ ] 多轮追问能复用历史 evidence，不重新全量调研
- [ ] LangSmith 中可查看完整 graph 执行轨迹（含并行节点）
- [ ] 评测集平均得分 ≥ 80（满分 100）
- [ ] `docker compose up` 一键启动，5 分钟内可在浏览器使用
- [ ] README 含架构图、Demo GIF、面试讲解要点

---

## 十一、面试讲解的核心亮点（作品集叙事）

1. **"用 Supervisor + 并行 sub-agent 模式实现 Deep Research"**
   相比线性 pipeline，并行调研让 N 个子问题同时执行，平均响应时间降低显著；Supervisor 用 LLM 做路由，遇到 fallback 时退回规则路由保证鲁棒性。

2. **"用 interrupt + Command(resume) 实现计划确认 HITL"**
   避免 agent 跑完才发现方向错了，节省 token 与等待时间；体现对 LangGraph 高级特性的掌握。

3. **"Reflector 反思循环 + 最大迭代兜底"**
   解决"agent 一次输出质量不可控"的问题，平衡质量与成本，对应面试常考的"Reflexion / Self-Critique"模式。

4. **"SqliteSaver + thread_id 支撑多轮对话"**
   会话状态可恢复，浏览器刷新不丢；不同用户会话隔离；完整体现 LangGraph 持久化机制。

5. **"自建评测集 + LLM-as-judge"**
   每次 prompt/路由策略改动都有量化对比，体现工程师而非"prompt 调参侠"的素养。

6. **"SSE 流式推送 agent 状态给前端"**
   用户能看到每个 agent 在干什么，体感流畅；体现对生产级 UX 的关注。

7. **"复用 01_RAG 的混合检索作为 KB Researcher"**
   项目间技术复用，体现工程化思维和系统设计能力。

---

## 十二、与其他子项目的关系

| 关系 | 说明 |
|---|---|
| **复用 01_RAG** | KB Researcher 直接复用 01 的混合检索（向量 + BM25 + RRF），知识库可共享 |
| **延展 02_REACT_AGENT** | 02 是单 agent + tool use，03 升级为 multi-agent + 协作 |
| **铺垫 04_HUMAN_IN_THE_LOOP** | 03 的 interrupt 是 04 深入的预热，04 会做更复杂的合同审核中断/恢复 |
| **避让 05_PRODUCT_AGENT** | 05 是完整生产级（Postgres / Redis / Prometheus / Mem0），03 定位"中等生产化"，避免重复 |

---

## 十三、扩展方向（v2 候选）

- 引入 **Mem0 长期记忆**，跨会话记住用户偏好（关注的厂商、不喜欢的信息源等）
- 支持 **多用户 + 鉴权**（FastAPI + JWT）
- 报告生成后接 **MCP Notion / 飞书** server 自动归档
- **并行加速**：sub-agent 用 asyncio.gather 真正并行调用 LLM
- 加入 **Cost Tracker**，每次执行展示 token / 成本
- 给 Web Researcher 接入 **Playwright** 做动态页面抓取
