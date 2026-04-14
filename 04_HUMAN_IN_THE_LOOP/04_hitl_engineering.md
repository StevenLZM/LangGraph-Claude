# 工程设计：Human-in-the-Loop — 智能合同审核系统

> 对应 PRD：04_human_in_the_loop_contract.md

---

## 一、整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    前端层（双界面）                              │
│  ┌──────────────────────────┐  ┌───────────────────────────┐   │
│  │   上传入口（Streamlit）   │  │  审核仪表盘（Streamlit）   │   │
│  │   - 上传合同文件          │  │  - 待审核列表             │   │
│  │   - 提交分析任务          │  │  - 详情查看 + 决策按钮    │   │
│  └─────────────┬────────────┘  └──────────┬────────────────┘   │
└────────────────┼──────────────────────────┼────────────────────┘
                 │ POST /analyze             │ POST /review/decision
┌────────────────▼──────────────────────────▼────────────────────┐
│                    FastAPI 后端                                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  /analyze 接口  │  │ /review 接口   │  │  /status 接口   │  │
│  │  (触发Graph执行) │  │(注入人工决策)  │  │(查询执行状态)   │  │
│  └────────┬────────┘  └───────┬────────┘  └─────────────────┘  │
└───────────┼───────────────────┼────────────────────────────────┘
            │                   │ Command(resume=decision)
┌───────────▼───────────────────▼────────────────────────────────┐
│                  LangGraph 工作流引擎                            │
│                                                                 │
│  [解析] → [提取] → [风险识别] → ⏸[人工审核] → [报告] → [通知]  │
│                                    ↑                           │
│                              interrupt()                       │
│                         SqliteSaver 持久化                     │
└─────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    基础设施层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  SQLite DB   │  │  MCP PDF解析 │  │   邮件/Webhook通知   │  │
│  │ (状态+日志)  │  │  Server      │  │                      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、LangGraph 核心设计

### 2.1 完整状态定义

```python
# graph/state.py
from typing import TypedDict, List, Annotated, Optional, Literal
from langgraph.graph.message import add_messages

class ContractInfo(TypedDict):
    party_a: str             # 甲方
    party_b: str             # 乙方
    amount: str              # 合同金额
    start_date: str          # 开始日期
    end_date: str            # 结束日期
    payment_terms: str       # 付款方式
    jurisdiction: str        # 争议管辖地

class RiskItem(TypedDict):
    clause_id: str           # 条款编号（如"第8条"）
    risk_type: str           # 风险类型
    severity: Literal["high", "medium", "low"]
    description: str         # 风险描述
    weight: int              # 权重（影响评分）
    suggestion: str          # 修改建议

class HumanDecision(TypedDict):
    action: Literal["approve", "modify", "reject"]
    comments: str            # 审核意见
    reviewer_id: str         # 审核人
    reviewed_at: str         # 审核时间

class ContractState(TypedDict):
    # 标识
    contract_id: str
    file_path: str

    # 解析结果
    raw_text: str
    page_count: int
    extracted_info: Optional[ContractInfo]

    # 风险分析
    risk_items: List[RiskItem]
    risk_score: int               # 0-100，越高风险越大
    ai_recommendation: str

    # 人工审核
    requires_human_review: bool   # 是否需要人工介入
    human_decision: Optional[HumanDecision]
    review_round: int             # 审核轮次（支持多轮）
    review_history: List[dict]    # 历史审核记录

    # 输出
    final_report: str
    notification_sent: bool
    audit_log: List[dict]         # 完整审计轨迹

    messages: Annotated[list, add_messages]
```

### 2.2 图结构与 Interrupt 机制

```python
# graph/workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

def build_contract_graph(db_path: str = "contracts.db"):
    workflow = StateGraph(ContractState)

    # 注册节点
    workflow.add_node("parser",         parse_document_node)
    workflow.add_node("extractor",      extract_info_node)
    workflow.add_node("risk_analyzer",  risk_analysis_node)
    workflow.add_node("human_review",   human_review_node)   # 关键：包含 interrupt
    workflow.add_node("ai_modifier",    ai_modify_node)      # AI 根据意见修改建议
    workflow.add_node("report_generator", report_node)
    workflow.add_node("notifier",       notify_node)

    # 边
    workflow.set_entry_point("parser")
    workflow.add_edge("parser",       "extractor")
    workflow.add_edge("extractor",    "risk_analyzer")
    workflow.add_edge("ai_modifier",  "human_review")  # 修改后重新审核
    workflow.add_edge("report_generator", "notifier")
    workflow.add_edge("notifier",     END)

    # 条件路由：风险评分决定是否需要人工
    workflow.add_conditional_edges(
        "risk_analyzer",
        route_by_risk_score,
        {
            "auto_approve":    "report_generator",  # 低风险自动通过
            "human_required":  "human_review",      # 高风险人工审核
        }
    )

    # 条件路由：人工决策后的走向
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_decision,
        {
            "approve":  "report_generator",
            "modify":   "ai_modifier",
            "reject":   "report_generator",   # 生成拒绝报告
        }
    )

    # 使用 SqliteSaver 持久化（重启可恢复）
    checkpointer = SqliteSaver.from_conn_string(db_path)
    return workflow.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
```

### 2.3 Interrupt 节点实现

```python
# graph/nodes/human_review.py
from langgraph.types import interrupt

async def human_review_node(state: ContractState) -> dict:
    """
    此节点会触发 interrupt，暂停 Graph 执行。
    Graph 状态被持久化到 SQLite，等待外部注入人工决策。
    """
    
    # 准备给审核人的信息摘要
    review_package = {
        "contract_id": state["contract_id"],
        "extracted_info": state["extracted_info"],
        "risk_score": state["risk_score"],
        "risk_items": state["risk_items"],
        "ai_recommendation": state["ai_recommendation"],
        "review_round": state["review_round"],
        # 如果是修改后重审，提供对比
        "previous_reviews": state.get("review_history", [])
    }
    
    # ⏸ 暂停！等待外部通过 Command(resume=...) 恢复
    # interrupt() 的返回值就是恢复时传入的 human_decision
    human_decision: HumanDecision = interrupt(review_package)
    
    # 记录审计日志
    audit_entry = {
        "event": "human_review_completed",
        "decision": human_decision["action"],
        "reviewer": human_decision["reviewer_id"],
        "timestamp": human_decision["reviewed_at"],
        "comments": human_decision.get("comments", "")
    }
    
    return {
        "human_decision": human_decision,
        "review_round": state["review_round"] + 1,
        "review_history": state.get("review_history", []) + [audit_entry],
        "audit_log": state.get("audit_log", []) + [audit_entry]
    }
```

---

## 三、FastAPI 后端设计

### 3.1 API 接口定义

```python
# api/main.py
from fastapi import FastAPI, BackgroundTasks, UploadFile
from langgraph.types import Command

app = FastAPI()

@app.post("/contracts/analyze")
async def analyze_contract(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    """上传合同，启动分析流程"""
    contract_id = str(uuid.uuid4())
    file_path = f"/tmp/contracts/{contract_id}.pdf"
    
    # 保存文件
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # 异步启动 Graph（遇到 interrupt 会暂停，不会阻塞）
    initial_state = {
        "contract_id": contract_id,
        "file_path": file_path,
        "review_round": 0,
        "review_history": [],
        "audit_log": [],
        "notification_sent": False,
    }
    config = {"configurable": {"thread_id": contract_id}}
    
    background_tasks.add_task(
        run_contract_graph, initial_state, config
    )
    
    return {"contract_id": contract_id, "status": "processing"}


@app.post("/contracts/{contract_id}/decision")
async def submit_human_decision(
    contract_id: str,
    decision: HumanDecisionRequest
):
    """提交人工审核决策，恢复 Graph 执行"""
    config = {"configurable": {"thread_id": contract_id}}
    
    human_decision = {
        "action": decision.action,          # approve/modify/reject
        "comments": decision.comments,
        "reviewer_id": decision.reviewer_id,
        "reviewed_at": datetime.now().isoformat()
    }
    
    # 恢复 Graph 执行，将人工决策注入
    result = await graph_app.ainvoke(
        Command(resume=human_decision),
        config=config
    )
    
    return {"status": "resumed", "decision": decision.action}


@app.get("/contracts/{contract_id}/status")
async def get_contract_status(contract_id: str):
    """查询合同处理状态（前端轮询用）"""
    config = {"configurable": {"thread_id": contract_id}}
    state = graph_app.get_state(config)
    
    # 判断当前状态
    if state.next:
        # 有下一步节点，可能在等待 interrupt
        next_node = state.next[0] if state.next else None
        status = "awaiting_human_review" if next_node == "human_review" else "processing"
    else:
        status = "completed"
    
    return {
        "contract_id": contract_id,
        "status": status,
        "risk_score": state.values.get("risk_score"),
        "review_round": state.values.get("review_round", 0),
        "risk_items": state.values.get("risk_items", [])
    }


@app.get("/contracts/pending")
async def list_pending_reviews():
    """获取所有待审核合同列表"""
    # 从 SQLite 中查询所有 interrupted 的 threads
    pending = await db.query(
        "SELECT thread_id FROM checkpoints WHERE status='interrupted'"
    )
    return {"pending": [p["thread_id"] for p in pending]}
```

### 3.2 状态查询与恢复

```python
# api/graph_manager.py

async def run_contract_graph(initial_state: dict, config: dict):
    """后台执行 Graph，遇到 interrupt 自动暂停"""
    try:
        result = await graph_app.ainvoke(initial_state, config=config)
        await update_contract_db(
            config["configurable"]["thread_id"],
            status="completed",
            result=result
        )
    except Exception as e:
        # Graph 在 interrupt 处暂停不是异常，是正常流程
        logger.info(f"Graph paused (possibly at interrupt): {e}")

def get_interrupted_state(contract_id: str) -> dict:
    """获取 interrupt 时的状态（用于展示给审核人）"""
    config = {"configurable": {"thread_id": contract_id}}
    state = graph_app.get_state(config)
    
    # interrupt 时，state.tasks 包含暂停节点的信息
    interrupted_task = state.tasks[0] if state.tasks else None
    if interrupted_task and interrupted_task.interrupts:
        # interrupt() 传入的数据（review_package）
        review_data = interrupted_task.interrupts[0].value
        return review_data
    return {}
```

---

## 四、各节点详细实现

### 4.1 风险识别节点

```python
# graph/nodes/risk_analyzer.py

RISK_ANALYSIS_PROMPT = """
你是一位资深法律顾问，专门识别合同中的风险条款。

分析以下合同文本，识别所有潜在风险：

【高风险项（权重3分）】
- 单方面变更权（一方可单方面修改合同内容）
- 无限免责条款（某方对任何损失不承担责任）
- 不对等违约责任（双方违约金差距超过5倍）
- 争议管辖在对方所在地

【中风险项（权重2分）】
- 模糊的交付标准（缺乏可量化的验收指标）
- 付款条件不清晰
- 知识产权归属不明确

【低风险项（权重1分）】
- 通知方式不够完善
- 保密期限过短或过长

风险评分 = min(各风险项权重之和 * 10, 100)

合同文本：
{contract_text}

提取的基本信息：
{extracted_info}

请输出JSON格式的风险报告：
{
  "risk_items": [...],
  "risk_score": 整数,
  "ai_recommendation": "综合审核建议（200字以内）"
}
"""

async def risk_analysis_node(state: ContractState) -> dict:
    chain = risk_prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({
        "contract_text": state["raw_text"][:6000],  # 避免超 token 限制
        "extracted_info": json.dumps(state["extracted_info"], ensure_ascii=False)
    })
    
    risk_score = result["risk_score"]
    requires_human = risk_score >= 30  # 30分以上需要人工审核
    
    return {
        "risk_items": result["risk_items"],
        "risk_score": risk_score,
        "ai_recommendation": result["ai_recommendation"],
        "requires_human_review": requires_human,
        "audit_log": state.get("audit_log", []) + [{
            "event": "risk_analysis_completed",
            "score": risk_score,
            "requires_human": requires_human,
            "timestamp": datetime.now().isoformat()
        }]
    }

def route_by_risk_score(state: ContractState) -> str:
    if state["risk_score"] < 30:
        return "auto_approve"
    return "human_required"
```

### 4.2 报告生成节点

```python
# graph/nodes/report_generator.py

REPORT_TEMPLATE = """
# 合同审核报告

**合同编号**: {contract_id}
**审核时间**: {review_time}
**最终决定**: {decision_emoji} {decision_text}

---

## 合同基本信息
| 项目 | 内容 |
|------|------|
| 甲方 | {party_a} |
| 乙方 | {party_b} |
| 金额 | {amount} |
| 期限 | {start_date} ~ {end_date} |

---

## 风险评估
**综合风险评分**: {risk_score}/100 {risk_level}

{risk_items_table}

---

## AI 审核建议
{ai_recommendation}

---

## 人工审核记录
{review_history_table}

---

## 最终审核意见
**审核人**: {reviewer_id}
**决定**: {decision_text}
**意见**: {comments}

---
*本报告由 AI 辅助生成，最终决定由人工审核员确认*
"""

async def report_node(state: ContractState) -> dict:
    decision = state.get("human_decision", {})
    action = decision.get("action", "auto_approve")
    
    decision_map = {
        "approve": ("✅", "批准通过"),
        "reject": ("❌", "拒绝"),
        "auto_approve": ("🤖", "AI自动通过（低风险）")
    }
    emoji, text = decision_map.get(action, ("❓", "未知"))
    
    report = REPORT_TEMPLATE.format(
        contract_id=state["contract_id"],
        review_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        decision_emoji=emoji,
        decision_text=text,
        **state["extracted_info"],
        risk_score=state["risk_score"],
        risk_level="🔴 高风险" if state["risk_score"] >= 60 else "🟡 中风险" if state["risk_score"] >= 30 else "🟢 低风险",
        risk_items_table=format_risk_table(state["risk_items"]),
        ai_recommendation=state["ai_recommendation"],
        review_history_table=format_review_history(state["review_history"]),
        reviewer_id=decision.get("reviewer_id", "系统"),
        comments=decision.get("comments", "无")
    )
    
    return {"final_report": report}
```

---

## 五、MCP 集成设计

### 5.1 PDF 解析 MCP Server

```python
# mcp_servers/pdf_server.py
from mcp.server import Server
from mcp.types import Tool, TextContent
import fitz  # PyMuPDF

server = Server("pdf-parser")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="parse_pdf",
            description="解析 PDF 文件，提取文本内容，保留页码信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "PDF文件的绝对路径"},
                    "max_pages": {"type": "integer", "default": 100}
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="extract_tables",
            description="从 PDF 中提取表格数据",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "page_number": {"type": "integer"}
                },
                "required": ["file_path"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "parse_pdf":
        doc = fitz.open(arguments["file_path"])
        pages = []
        for i, page in enumerate(doc):
            if i >= arguments.get("max_pages", 100):
                break
            text = page.get_text()
            pages.append(f"=== 第{i+1}页 ===\n{text}")
        
        full_text = "\n\n".join(pages)
        return [TextContent(type="text", text=full_text)]
```

### 5.2 MCP 配置

```json
{
  "mcpServers": {
    "pdf-parser": {
      "command": "python",
      "args": ["-m", "mcp_servers.pdf_server"],
      "description": "PDF 解析服务"
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/tmp/contracts", "/output/reports"],
      "description": "合同文件和报告的文件系统访问"
    },
    "email": {
      "command": "python",
      "args": ["-m", "mcp_servers.email_server"],
      "env": {
        "SMTP_HOST": "${SMTP_HOST}",
        "SMTP_USER": "${SMTP_USER}",
        "SMTP_PASS": "${SMTP_PASS}"
      },
      "description": "发送审核通知邮件"
    }
  }
}
```

---

## 六、Streamlit 审核界面设计

```python
# pages/reviewer_dashboard.py
import streamlit as st
import requests, time

def render_review_dashboard():
    st.title("合同审核仪表盘")
    
    # 获取待审核列表
    pending = requests.get("http://localhost:8000/contracts/pending").json()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("待审核", len(pending["pending"]))
    
    # 选择要审核的合同
    selected = st.selectbox("选择合同", pending["pending"])
    
    if selected:
        # 获取 interrupt 时暂停的状态数据
        status = requests.get(f"http://localhost:8000/contracts/{selected}/status").json()
        review_data = requests.get(f"http://localhost:8000/contracts/{selected}/review-data").json()
        
        # 展示合同摘要
        info = review_data.get("extracted_info", {})
        st.subheader("合同摘要")
        col1, col2 = st.columns(2)
        col1.write(f"**甲方**: {info.get('party_a', '未知')}")
        col1.write(f"**金额**: {info.get('amount', '未知')}")
        col2.write(f"**乙方**: {info.get('party_b', '未知')}")
        col2.write(f"**期限**: {info.get('start_date')} ~ {info.get('end_date')}")
        
        # 风险评分
        risk_score = review_data.get("risk_score", 0)
        color = "red" if risk_score >= 60 else "orange" if risk_score >= 30 else "green"
        st.markdown(f"### 风险评分: :{color}[{risk_score}/100]")
        
        # 风险项详情
        st.subheader("风险项")
        for item in review_data.get("risk_items", []):
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[item["severity"]]
            with st.expander(f"{severity_icon} {item['clause_id']}: {item['risk_type']}"):
                st.write(item["description"])
                st.info(f"建议: {item['suggestion']}")
        
        # AI 建议
        st.subheader("AI 审核建议")
        st.write(review_data.get("ai_recommendation", ""))
        
        # 审核操作
        st.subheader("审核决策")
        comments = st.text_area("审核意见（必填）", placeholder="请输入您的审核意见...")
        
        col1, col2, col3 = st.columns(3)
        
        if col1.button("✅ 批准通过", type="primary"):
            if not comments:
                st.error("请输入审核意见")
            else:
                submit_decision(selected, "approve", comments)
        
        if col2.button("✏️ 要求修改"):
            if not comments:
                st.error("请输入需要修改的具体内容")
            else:
                submit_decision(selected, "modify", comments)
        
        if col3.button("❌ 拒绝", type="secondary"):
            if not comments:
                st.error("请输入拒绝原因")
            else:
                submit_decision(selected, "reject", comments)

def submit_decision(contract_id: str, action: str, comments: str):
    response = requests.post(
        f"http://localhost:8000/contracts/{contract_id}/decision",
        json={
            "action": action,
            "comments": comments,
            "reviewer_id": st.session_state.get("reviewer_id", "reviewer_001")
        }
    )
    if response.status_code == 200:
        st.success(f"决策已提交: {action}")
        st.rerun()
```

---

## 七、审计日志设计

```python
# db/audit_log.py
import sqlite3
from datetime import datetime

def init_audit_db(db_path: str = "audit.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,
            event       TEXT NOT NULL,        -- risk_analyzed / human_review / approved / ...
            actor       TEXT,                 -- 操作人（AI/审核人ID）
            details     TEXT,                 -- JSON 格式的详情
            timestamp   TEXT NOT NULL
        )
    """)
    conn.commit()

def log_event(contract_id: str, event: str, actor: str, details: dict):
    conn = sqlite3.connect("audit.db")
    conn.execute(
        "INSERT INTO audit_log VALUES (NULL, ?, ?, ?, ?, ?)",
        (contract_id, event, actor, json.dumps(details), datetime.now().isoformat())
    )
    conn.commit()
```

---

## 八、目录结构

```
04_hitl_contract/
├── app.py                        # Streamlit 入口
├── pages/
│   ├── upload.py                 # 上传页
│   └── reviewer_dashboard.py    # 审核仪表盘
├── api/
│   ├── main.py                   # FastAPI 路由
│   ├── schemas.py                # 请求/响应模型
│   └── graph_manager.py         # Graph 调用封装
├── graph/
│   ├── state.py                  # 状态定义
│   ├── workflow.py               # 图构建
│   └── nodes/
│       ├── parser.py             # 文档解析节点
│       ├── extractor.py          # 信息提取节点
│       ├── risk_analyzer.py      # 风险分析节点
│       ├── human_review.py       # 人工审核节点（含interrupt）
│       ├── ai_modifier.py        # AI修改建议节点
│       ├── report_generator.py   # 报告生成节点
│       └── notifier.py           # 通知节点
├── mcp_servers/
│   ├── pdf_server.py             # PDF解析 MCP Server
│   └── email_server.py          # 邮件通知 MCP Server
├── db/
│   └── audit_log.py             # 审计日志
├── .mcp.json
├── contracts.db                  # LangGraph 状态持久化
├── audit.db                      # 审计日志数据库
└── requirements.txt
```
