# 工程设计：多 Agent 协作系统 — AI 内容创作团队

> 对应 PRD：03_multi_agent_content_team.md

---

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  [主题输入] [执行进度] [各Agent输出] [最终文章预览] [导出]        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                   LangGraph 工作流引擎                           │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │Researcher│→│  Writer  │→│ Reviewer │→│SEO Specialist│   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │    Agent     │   │
│  └──────────┘  └──────────┘  └────┬─────┘  └──────────────┘   │
│                      ↑            │ 不通过                      │
│                      └────────────┘ (修改循环)                  │
│                                                                 │
│  共享状态 (ArticleState) ←→ LangSmith Tracing                   │
└─────────────────────────────────────────────────────────────────┘
                            │ tools
┌───────────────────────────▼─────────────────────────────────────┐
│                       工具 & MCP 层                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Tavily搜索 │  │ArXiv论文   │  │ MCP Brave  │  │文件系统  │  │
│  │  (实时信息)│  │ (学术引用) │  │  搜索增强  │  │MCP Server│  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、LangGraph 状态与图设计

### 2.1 完整状态定义

```python
# graph/state.py
from typing import TypedDict, List, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class ResearchReport(TypedDict):
    key_points: List[str]          # 核心要点列表
    references: List[str]          # 参考资料 URL
    outline: List[str]             # 建议文章结构
    background: str                # 背景信息摘要

class ReviewResult(TypedDict):
    score: int                     # 综合评分 0-100
    passed: bool                   # 是否通过（>=85）
    accuracy_score: int            # 准确性
    readability_score: int         # 可读性
    completeness_score: int        # 完整性
    issues: List[str]              # 发现的问题
    suggestions: List[str]         # 改进建议

class ArticleState(TypedDict):
    # === 输入 ===
    topic: str                     # 文章主题
    audience: str                  # 受众（beginner/intermediate/expert）
    target_words: int              # 目标字数
    style: str                     # 写作风格（tutorial/analysis/news）

    # === 调研阶段 ===
    research_report: Optional[ResearchReport]

    # === 写作阶段 ===
    draft: str                     # 当前草稿
    draft_version: int             # 版本号

    # === 审校阶段 ===
    review_result: Optional[ReviewResult]
    revision_count: int            # 修改次数（上限3次）
    revision_history: List[dict]   # 历次修改记录 [{version, score, issues}]

    # === SEO 阶段 ===
    seo_keywords: List[str]
    final_article: str
    meta_description: str          # SEO 摘要（150字以内）

    # === 执行追踪 ===
    messages: Annotated[list[BaseMessage], add_messages]
    current_node: str              # 当前执行节点（UI展示用）
    start_time: float
    total_tokens_used: int
```

### 2.2 图结构定义

```python
# graph/workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_article_graph():
    workflow = StateGraph(ArticleState)

    # === 注册节点 ===
    workflow.add_node("researcher",   researcher_node)
    workflow.add_node("writer",       writer_node)
    workflow.add_node("reviewer",     reviewer_node)
    workflow.add_node("reviser",      reviser_node)
    workflow.add_node("seo",          seo_node)

    # === 定义边 ===
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer",     "reviewer")
    workflow.add_edge("reviser",    "reviewer")  # 修改后重新审校
    workflow.add_edge("seo",        END)

    # === 条件路由 ===
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "approve":      "seo",          # 评分通过 → SEO优化
            "revise":       "reviser",      # 需要修改 → 修改节点
            "force_pass":   "seo",          # 超过最大修改次数 → 强制通过
        }
    )

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

def route_after_review(state: ArticleState) -> str:
    """审校路由逻辑"""
    review = state["review_result"]
    revision_count = state["revision_count"]
    
    if revision_count >= 3:
        return "force_pass"  # 超过上限，强制通过
    if review["passed"]:
        return "approve"
    return "revise"
```

---

## 三、各 Agent 节点详细设计

### 3.1 调研 Agent (Researcher)

```python
# agents/researcher.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearchResults

RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的技术调研专家。
你的任务是为文章主题收集全面、准确的背景资料。

输出格式（严格按JSON输出）：
{{
  "key_points": ["核心要点1", "核心要点2", ...],  // 5-8个要点
  "references": ["URL1", "URL2", ...],            // 3-5个来源
  "outline": ["第一章", "第二章", ...],            // 建议文章结构
  "background": "背景信息摘要（200字以内）"
}}"""),
    ("human", """
主题：{topic}
受众：{audience}
目标字数：{target_words}字

请先搜索相关资料，然后整理调研报告。
""")
])

async def researcher_node(state: ArticleState) -> dict:
    """调研节点：搜索资料，生成调研报告"""
    
    # 构建搜索查询
    search_queries = [
        state["topic"],
        f"{state['topic']} 教程 最新",
        f"{state['topic']} 原理 详解",
    ]
    
    # 并行搜索
    search_tool = TavilySearchResults(max_results=3)
    search_results = []
    for query in search_queries:
        results = await search_tool.ainvoke(query)
        search_results.extend(results)
    
    # 整合搜索结果
    context = "\n\n".join([
        f"来源: {r['url']}\n内容: {r['content'][:500]}"
        for r in search_results[:6]
    ])
    
    # LLM 生成调研报告
    chain = RESEARCHER_PROMPT | llm | JsonOutputParser()
    report = await chain.ainvoke({
        "topic": state["topic"],
        "audience": state["audience"],
        "target_words": state["target_words"],
        "search_results": context
    })
    
    return {
        "research_report": report,
        "current_node": "researcher_done",
        "messages": [AIMessage(content=f"调研完成，发现 {len(report['key_points'])} 个核心要点")]
    }
```

### 3.2 写作 Agent (Writer)

```python
# agents/writer.py

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位资深技术写作专家，擅长将复杂技术内容写得清晰易懂。

写作要求：
- 按照提供的文章结构撰写完整内容
- 受众是{audience}，相应调整技术深度
- 目标字数：{target_words}字（±20%）
- 格式：Markdown，包含标题、代码块、重点标注
- 代码示例要完整可运行
- 每个概念配合具体例子

{revision_guidance}
"""),
    ("human", """
主题：{topic}
调研报告：{research_report}
文章结构：{outline}

{revision_feedback}

请撰写完整文章：
""")
])

async def writer_node(state: ArticleState) -> dict:
    """写作节点：基于调研报告生成/修改文章"""
    
    # 判断是初始写作还是修改
    is_revision = state["draft_version"] > 0
    revision_guidance = ""
    revision_feedback = ""
    
    if is_revision and state["review_result"]:
        review = state["review_result"]
        revision_guidance = f"""
【修改指导】
上一版本评分：{review['score']}/100
需要改进的问题：
{chr(10).join(f'- {issue}' for issue in review['issues'])}
改进建议：
{chr(10).join(f'- {sug}' for sug in review['suggestions'])}
"""
        revision_feedback = f"请基于以上反馈修改文章，当前草稿：\n{state['draft']}"
    
    chain = WRITER_PROMPT | llm | StrOutputParser()
    draft = await chain.ainvoke({
        "topic": state["topic"],
        "audience": state["audience"],
        "target_words": state["target_words"],
        "research_report": json.dumps(state["research_report"], ensure_ascii=False),
        "outline": "\n".join(state["research_report"]["outline"]),
        "revision_guidance": revision_guidance,
        "revision_feedback": revision_feedback
    })
    
    return {
        "draft": draft,
        "draft_version": state["draft_version"] + 1,
        "current_node": "writer_done"
    }
```

### 3.3 审校 Agent (Reviewer)

```python
# agents/reviewer.py

REVIEWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位严格的技术文章审校专家。
从以下3个维度评审文章，每个维度0-100分，计算加权总分：

评分维度：
1. 准确性 (40%权重)：技术内容是否正确，代码是否可运行
2. 可读性 (30%权重)：结构是否清晰，表达是否流畅
3. 完整性 (30%权重)：是否覆盖主题核心内容，字数是否达标

通过标准：加权总分 ≥ 85

输出严格按JSON格式：
{{
  "accuracy_score": 整数,
  "readability_score": 整数,
  "completeness_score": 整数,
  "score": 加权总分整数,
  "passed": true/false,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}}
"""),
    ("human", """
主题：{topic}
目标受众：{audience}
目标字数：{target_words}

待审校文章：
{draft}

请给出严格的评审结果：
""")
])

async def reviewer_node(state: ArticleState) -> dict:
    chain = REVIEWER_PROMPT | llm | JsonOutputParser()
    review = await chain.ainvoke({
        "topic": state["topic"],
        "audience": state["audience"],
        "target_words": state["target_words"],
        "draft": state["draft"]
    })
    
    # 更新历史记录
    history_entry = {
        "version": state["draft_version"],
        "score": review["score"],
        "issues": review["issues"]
    }
    
    return {
        "review_result": review,
        "revision_count": state["revision_count"] + (0 if review["passed"] else 1),
        "revision_history": state.get("revision_history", []) + [history_entry],
        "current_node": "reviewer_done"
    }
```

### 3.4 SEO Agent

```python
# agents/seo.py

SEO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位 SEO 优化专家，专注于技术内容的搜索引擎优化。

优化任务：
1. 优化文章标题（包含主要关键词，吸引点击）
2. 提取 5-10 个核心关键词
3. 生成 Meta Description（150字以内，包含关键词）
4. 在文章自然位置插入关键词（不影响阅读体验）
5. 检查标题层级结构（H1只有一个，H2/H3合理分布）

输出JSON格式：
{{
  "optimized_title": "优化后的标题",
  "keywords": ["关键词1", ...],
  "meta_description": "150字以内的摘要",
  "final_article": "完整优化后的文章Markdown"
}}
"""),
    ("human", "主题: {topic}\n\n原文章:\n{draft}")
])

async def seo_node(state: ArticleState) -> dict:
    chain = SEO_PROMPT | llm | JsonOutputParser()
    result = await chain.ainvoke({
        "topic": state["topic"],
        "draft": state["draft"]
    })
    return {
        "seo_keywords": result["keywords"],
        "final_article": result["final_article"],
        "meta_description": result["meta_description"],
        "current_node": "completed"
    }
```

---

## 四、MCP 集成设计

### 4.1 MCP Server 配置

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"},
      "description": "Brave 搜索，作为 Tavily 的补充搜索源"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/output/articles"],
      "description": "文章输出目录，用于保存生成的文章"
    },
    "fetch": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"],
      "description": "抓取指定 URL 内容，用于获取参考文章"
    }
  }
}
```

### 4.2 Researcher 使用 MCP 搜索

```python
# 调研 Agent 同时使用 Tavily + MCP Brave Search
async def multi_source_search(query: str) -> list[dict]:
    results = []
    
    # 来源1: Tavily（主要）
    tavily_results = await tavily_client.search(query)
    results.extend(tavily_results["results"])
    
    # 来源2: MCP Brave Search（补充）
    brave_results = await mcp_client.call_tool(
        "brave-search", "brave_web_search",
        {"query": query, "count": 3}
    )
    results.extend(brave_results)
    
    # 来源3: MCP Fetch 获取具体页面内容
    for url in [r["url"] for r in results[:2]]:
        page_content = await mcp_client.call_tool(
            "fetch", "fetch",
            {"url": url, "max_length": 2000}
        )
        results.append({"url": url, "content": page_content, "source": "fetch"})
    
    return results

# 完成后通过 MCP 保存文章
async def save_article_via_mcp(article: str, title: str):
    filename = f"{title.replace(' ', '_')[:50]}.md"
    await mcp_client.call_tool(
        "filesystem", "write_file",
        {"path": f"/output/articles/{filename}", "content": article}
    )
```

---

## 五、Prompt 工程设计

### 5.1 受众自适应策略

```python
AUDIENCE_CONFIGS = {
    "beginner": {
        "description": "初学者，无相关背景知识",
        "requirements": "避免术语，多用类比，每个概念都要解释，代码要有详细注释",
        "code_style": "step-by-step，每行都有注释",
    },
    "intermediate": {
        "description": "有一定基础，了解基本概念",
        "requirements": "可使用常见术语，重点放在实战技巧和最佳实践",
        "code_style": "完整示例，关键处注释",
    },
    "expert": {
        "description": "资深工程师，追求深度",
        "requirements": "直接深入细节，分析原理和权衡，引用最新研究和进展",
        "code_style": "精简高效，展示高级用法",
    }
}
```

### 5.2 修改反馈 Prompt 策略

```python
def build_revision_context(state: ArticleState) -> str:
    """根据历史审校记录生成修改指导"""
    history = state.get("revision_history", [])
    if not history:
        return ""
    
    # 展示评分趋势
    scores = [h["score"] for h in history]
    trend = "↑提升" if len(scores) > 1 and scores[-1] > scores[-2] else "↓下降"
    
    last_review = state["review_result"]
    return f"""
【修改历史】第{len(history)}次修改，分数趋势: {' → '.join(map(str, scores))} {trend}

【当前问题（必须解决）】
{chr(10).join(f'{i+1}. {issue}' for i, issue in enumerate(last_review['issues']))}

【改进建议】
{chr(10).join(f'- {sug}' for sug in last_review['suggestions'])}

注意：这是第{state['revision_count']}次修改，最多允许3次，请务必解决所有问题。
"""
```

---

## 六、LangSmith 可观测性配置

```python
# config/tracing.py
import os
from langsmith import Client

def setup_tracing():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "multi-agent-article"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# 为每个 Agent 添加运行标签
from langchain_core.callbacks import LangChainTracer

def get_tracer(agent_name: str, run_id: str):
    return LangChainTracer(
        tags=[f"agent:{agent_name}", f"run:{run_id}"],
        metadata={"project": "article-generation"}
    )
```

---

## 七、目录结构

```
03_multi_agent_content/
├── app.py
├── graph/
│   ├── state.py              # ArticleState 定义
│   ├── workflow.py           # LangGraph 图构建
│   └── router.py             # 条件路由函数
├── agents/
│   ├── researcher.py         # 调研 Agent
│   ├── writer.py             # 写作 Agent
│   ├── reviewer.py           # 审校 Agent
│   ├── reviser.py            # 修改 Agent（复用 Writer）
│   └── seo.py                # SEO Agent
├── tools/
│   ├── search.py             # 搜索工具封装
│   └── mcp_tools.py          # MCP 工具加载
├── prompts/
│   └── templates.py          # 所有 Prompt 模板
├── config/
│   ├── tracing.py            # LangSmith 配置
│   └── audience.py           # 受众配置
├── .mcp.json
├── .env.example
└── requirements.txt
```

---

## 八、执行追踪与调试

```python
# 执行时打印完整状态变化
async def run_with_debug(topic: str, audience: str, target_words: int):
    initial_state = {
        "topic": topic,
        "audience": audience,
        "target_words": target_words,
        "style": "tutorial",
        "draft_version": 0,
        "revision_count": 0,
        "revision_history": [],
        "messages": [],
        "current_node": "start",
        "start_time": time.time(),
        "total_tokens_used": 0
    }
    
    async for event in app.astream(initial_state, stream_mode="values"):
        node = event.get("current_node", "unknown")
        
        if node == "researcher_done":
            print(f"✅ 调研完成: {len(event['research_report']['key_points'])} 个要点")
        elif node == "writer_done":
            print(f"✅ 写作完成: 第{event['draft_version']}稿, {len(event['draft'])}字")
        elif node == "reviewer_done":
            review = event["review_result"]
            status = "✅ 通过" if review["passed"] else "❌ 需修改"
            print(f"{status}: 评分 {review['score']}/100")
        elif node == "completed":
            elapsed = time.time() - event["start_time"]
            print(f"🎉 完成! 耗时 {elapsed:.1f}秒")
```
