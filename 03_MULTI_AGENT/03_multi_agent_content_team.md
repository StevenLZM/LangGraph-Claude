# PRD：多 Agent 协作系统 — AI 内容创作团队

> 项目编号：03 | 难度：⭐⭐⭐⭐ | 预计周期：3 周

---

## 一、项目背景与目标

### 背景

复杂任务（如撰写一篇深度技术文章）往往需要多个角色协作：调研员收集资料、写作者起草内容、审校者检查质量、SEO 专家优化关键词。单一 Agent 难以在一次对话中完成所有角色的职责，且容易注意力分散、质量不稳定。

### 目标

使用 LangGraph 构建一个多 Agent 协作的内容创作系统，每个 Agent 专注于特定角色，通过图状工作流协调执行，最终产出高质量的技术文章。

### 学习目标

| 技能点 | 掌握内容 |
|--------|----------|
| LangGraph 核心 | StateGraph、Node、Edge 的定义与组合 |
| 状态管理 | 跨 Agent 共享状态的设计与更新 |
| 角色分工 | 每个 Agent 的 System Prompt 设计 |
| 条件路由 | 根据状态动态决定下一个执行节点 |
| 循环控制 | 审校不通过时的修改迭代循环 |
| 消息传递 | Agent 间结构化数据传递格式 |

---

## 二、用户故事

```
作为一名技术博主
我想输入一个主题（如"LangGraph 入门教程"）
系统自动启动多个 AI 专家协作
完成调研、写作、审校、SEO 优化的全流程
最终输出一篇可发布的技术文章
而不是我手动重复调用 AI 多次整合结果
```

---

## 三、Agent 角色定义

### 3.1 调研 Agent（Researcher）

**职责**：根据主题搜索相关资料，整理成结构化的调研报告

**工具**：网络搜索、维基百科、ArXiv 论文搜索

**输出**：
```json
{
  "key_points": ["要点1", "要点2"],
  "references": ["来源URL1", "来源URL2"],
  "outline_suggestion": "建议文章结构"
}
```

### 3.2 写作 Agent（Writer）

**职责**：基于调研报告，按照提纲撰写完整文章

**输入**：调研报告 + 目标字数 + 受众定位

**输出**：Markdown 格式的完整文章草稿

### 3.3 审校 Agent（Reviewer）

**职责**：从准确性、可读性、结构完整性三个维度评审文章

**输出**：
```json
{
  "score": 85,
  "passed": false,
  "issues": ["第3段逻辑跳跃", "缺少代码示例"],
  "suggestions": ["添加实际代码演示", "补充背景知识"]
}
```

**通过标准**：综合评分 ≥ 85 分

### 3.4 SEO 优化 Agent（SEO Specialist）

**职责**：优化标题、添加关键词、调整摘要，提升搜索可见性

**输出**：SEO 优化后的最终文章 + 关键词列表

---

## 四、工作流设计

```
                    ┌─────────────┐
                    │  用户输入主题  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  调研 Agent  │
                    │  (搜索资料)  │
                    └──────┬──────┘
                           │ 调研报告
                           ▼
                    ┌─────────────┐
                    │  写作 Agent  │
                    │  (生成草稿)  │
                    └──────┬──────┘
                           │ 文章草稿
                           ▼
                    ┌─────────────┐
                    │  审校 Agent  │◄──────┐
                    │  (质量评审)  │       │
                    └──────┬──────┘       │
                           │              │
              ┌────────────┴───────────┐  │
              │ score >= 85?           │  │
              │                        │  │
           Yes│                     No │  │
              ▼                        ▼  │
    ┌─────────────┐          ┌─────────────┐
    │  SEO 优化   │          │  写作 Agent  │
    │   Agent     │          │  (根据反馈   │──┘
    └──────┬──────┘          │   修改文章) │
           │                 └─────────────┘
           ▼
    ┌─────────────┐
    │   最终输出   │
    └─────────────┘
```

---

## 五、功能需求

### 5.1 任务配置

- **F01** 用户输入主题、目标字数（500-5000字）、受众定位（初学者/中级/专家）
- **F02** 可选择跳过某个 Agent（如直接提供已有提纲，跳过调研）
- **F03** 设置审校最大迭代次数（默认 3 次），超过则强制输出

### 5.2 执行过程

- **F04** 实时展示当前执行节点和状态
- **F05** 每个 Agent 完成后展示其输出摘要
- **F06** 审校不通过时，展示具体问题和修改建议
- **F07** 记录完整执行日志，包括每次迭代的评分变化

### 5.3 结果输出

- **F08** 最终文章支持 Markdown 渲染预览
- **F09** 支持导出为 .md 文件
- **F10** 展示文章元数据：字数、关键词、SEO 评分、生成耗时

---

## 六、技术架构

### 状态定义

```python
from typing import TypedDict, List, Annotated
from langgraph.graph import add_messages

class ArticleState(TypedDict):
    topic: str                    # 用户输入的主题
    audience: str                 # 受众定位
    target_words: int             # 目标字数
    research_report: dict         # 调研报告
    draft: str                    # 文章草稿（当前版本）
    review_result: dict           # 最新审校结果
    revision_count: int           # 修改次数
    final_article: str            # 最终文章
    seo_keywords: List[str]       # SEO 关键词
    messages: Annotated[list, add_messages]  # 消息历史
```

### 图构建

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(ArticleState)

# 添加节点
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)
workflow.add_node("reviser", reviser_agent)   # 修改节点
workflow.add_node("seo", seo_agent)

# 定义边
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviser", "reviewer")  # 修改后重新审校

# 条件路由
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,   # 根据评分决定走向
    {
        "approved": "seo",
        "revision_needed": "reviser",
        "max_iterations": "seo"  # 超过最大次数强制通过
    }
)
workflow.add_edge("seo", END)
```

### 技术选型

| 层次 | 技术 |
|------|------|
| 工作流编排 | LangGraph StateGraph |
| LLM | claude-sonnet​-4-6（各 Agent 可配置不同模型）|
| 搜索工具 | Tavily API |
| 状态持久化 | LangGraph MemorySaver（内存）|
| UI | Streamlit |
| 可观测性 | LangSmith（追踪每个节点执行）|

---

## 七、评估标准

- [ ] 完整执行 researcher → writer → reviewer → seo 链路
- [ ] 审校评分 < 85 时触发修改循环，最多 3 次
- [ ] 第 3 次迭代后无论评分如何都能正常输出
- [ ] LangSmith 中可查看完整的 Graph 执行轨迹
- [ ] 生成一篇 1000 字以上、结构完整的技术文章

---

## 八、项目交付物

1. `app.py` — 主程序入口
2. `graph/state.py` — 状态定义
3. `graph/workflow.py` — LangGraph 图构建
4. `agents/researcher.py` — 调研 Agent
5. `agents/writer.py` — 写作 Agent
6. `agents/reviewer.py` — 审校 Agent
7. `agents/seo.py` — SEO 优化 Agent
8. `README.md` — 架构图 + 演示视频

---

## 九、扩展方向

- 引入监督 Agent（Supervisor）动态分配任务给子 Agent
- 支持并行执行（多个写作 Agent 同时生成不同章节）
- 添加人工审核节点（Human-in-the-loop）
- 支持多语言输出（中文/英文）
- 接入内容发布 API（掘金、知乎等平台）
