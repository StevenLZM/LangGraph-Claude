# PRD：Human-in-the-Loop Agent — 智能合同审核系统

> 项目编号：04 | 难度：⭐⭐⭐⭐ | 预计周期：2-3 周

---

## 一、项目背景与目标

### 背景

在高风险业务场景中（合同签署、资金审批、内容发布），AI 的自动执行可能带来无法挽回的损失。Human-in-the-Loop（HITL）模式在 Agent 执行关键步骤前暂停、等待人工确认，在保留 AI 效率的同时引入人类判断。

### 目标

构建一个合同智能审核系统：AI Agent 自动完成信息提取、风险识别、条款分析，在关键决策点（高风险条款、异常内容）暂停执行并通知人工审核，人工批准或修改后 Agent 继续后续流程。

### 学习目标

| 技能点 | 掌握内容 |
|--------|----------|
| LangGraph Interrupt | `interrupt()` 实现执行暂停 |
| Checkpoint | MemorySaver / SqliteSaver 状态持久化 |
| 状态恢复 | 从断点恢复 Graph 执行 |
| 异步等待 | Agent 暂停期间不阻塞其他任务 |
| 审核界面 | 人工介入的交互 UI 设计 |
| 审计日志 | 记录人工决策的完整痕迹 |

---

## 二、用户故事

```
作为一名法务专员
我想让 AI 自动完成合同的初步分析
当 AI 发现高风险条款或异常内容时
系统暂停并通知我进行人工复核
我可以选择：批准继续 / 要求修改 / 拒绝合同
AI 根据我的决策继续完成剩余流程
从而将我从重复性阅读工作中解放出来
专注于真正需要判断力的部分
```

---

## 三、工作流设计

```
合同上传
    │
    ▼
┌────────────────┐
│  文档解析节点   │  — 提取文本、元数据
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  信息提取节点   │  — 提取甲乙方、金额、期限、关键条款
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  风险识别节点   │  — 标记高风险条款、计算风险评分
└───────┬────────┘
        │
        ├── 风险评分 < 30（低风险）
        │         └──→ 跳过人工审核
        │
        └── 风险评分 ≥ 30（中高风险）
                  │
                  ▼
        ┌────────────────┐
        │  ⏸ 人工审核节点  │  ← INTERRUPT 暂停
        │  等待人工决策    │
        └───────┬────────┘
                │
    ┌───────────┼──────────────┐
    │           │              │
  批准继续    要求修改        拒绝
    │           │              │
    ▼           ▼              ▼
┌───────┐  ┌────────┐  ┌──────────┐
│报告生成│  │AI修改  │  │ 生成拒绝 │
│  节点  │  │建议节点│  │  报告    │
└───┬───┘  └───┬────┘  └──────────┘
    │           │
    ▼           └──→ 重新进入人工审核
┌────────┐
│发送通知 │  — 邮件/钉钉通知相关方
└────────┘
```

---

## 四、功能需求

### 4.1 自动分析阶段

- **F01** 支持上传 PDF/Word 格式合同文件
- **F02** 自动提取：甲方/乙方、合同金额、起止日期、付款方式、违约条款
- **F03** 自动识别风险项，包括但不限于：
  - 单方面解约权条款
  - 免责条款过于宽泛
  - 不合理的违约金比例
  - 模糊的交付标准
  - 争议管辖地不当
- **F04** 计算综合风险评分（0-100），标注每个风险项的权重

### 4.2 人工审核阶段

- **F05** 风险评分 ≥ 30 时自动触发人工审核流程
- **F06** 向审核人展示：合同摘要、风险项列表、AI 的审核建议
- **F07** 审核人可执行三种操作：
  - **批准**：确认 AI 分析无误，继续后续流程
  - **修改**：添加批注和修改意见，AI 重新生成建议
  - **拒绝**：终止合同流程，记录拒绝原因
- **F08** 审核等待期间，系统状态持久化（重启不丢失）
- **F09** 支持多人审核（串行：A 审后发给 B；并行：A 和 B 同时审）

### 4.3 后续流程

- **F10** 批准后生成标准化审核报告（PDF）
- **F11** 发送通知（邮件/Webhook）给相关方
- **F12** 记录完整审计日志：谁在什么时间做了什么决定

---

## 五、技术架构

### 核心：LangGraph Interrupt 机制

```python
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

def human_review_node(state: ContractState):
    """执行到此节点时暂停，等待人工输入"""
    
    # 将当前状态暴露给前端
    review_request = {
        "contract_summary": state["summary"],
        "risk_items": state["risk_items"],
        "risk_score": state["risk_score"],
        "ai_recommendation": state["ai_recommendation"]
    }
    
    # interrupt() 暂停执行，等待外部输入
    human_decision = interrupt(review_request)
    
    # 恢复执行后，human_decision 包含人工决策
    return {
        "human_decision": human_decision["action"],  # approve/modify/reject
        "human_comments": human_decision.get("comments", ""),
        "reviewer_id": human_decision["reviewer_id"]
    }
```

### 状态持久化

```python
# 使用 SqliteSaver 实现持久化，重启不丢失审核状态
checkpointer = SqliteSaver.from_conn_string("contracts.db")

app = workflow.compile(checkpointer=checkpointer)

# 每个合同有唯一的 thread_id
config = {"configurable": {"thread_id": f"contract_{contract_id}"}}

# 触发执行（遇到 interrupt 会暂停）
result = app.invoke(initial_state, config=config)

# 人工审核后，携带决策恢复执行
result = app.invoke(
    Command(resume=human_decision),
    config=config
)
```

### 状态定义

```python
class ContractState(TypedDict):
    # 输入
    contract_id: str
    file_path: str
    
    # 解析结果
    raw_text: str
    extracted_info: dict       # 甲乙方、金额、期限等
    
    # 风险分析
    risk_items: List[dict]     # 每个风险项详情
    risk_score: int            # 0-100
    ai_recommendation: str     # AI 建议
    
    # 人工审核
    human_decision: str        # approve / modify / reject
    human_comments: str        # 人工批注
    reviewer_id: str           # 审核人 ID
    review_history: List[dict] # 历史审核记录（支持多轮）
    
    # 输出
    final_report: str          # 最终审核报告
    notification_sent: bool
```

### 技术选型

| 层次 | 技术 |
|------|------|
| 工作流 | LangGraph with Interrupt |
| 状态持久化 | SqliteSaver（生产用 PostgresSaver）|
| LLM | claude-sonnet​​-4-6 |
| 文档解析 | PyMuPDF + python-docx |
| 前端 | Streamlit（实时轮询审核状态）|
| 通知 | SMTP 邮件 / Webhook |
| 审计日志 | SQLite 表记录每次决策 |

---

## 六、界面设计

### 仪表盘页面

```
合同审核系统
├── 待审核 (3)     ← 等待人工介入的合同
├── 审核中 (1)     ← 当前正在审核
├── 已完成 (12)    ← 审核结束的历史
└── 已拒绝 (2)
```

### 审核详情页

```
合同：《软件开发服务合同》v2.1   [风险评分: 67 ⚠️ 中高风险]

┌── 合同摘要 ─────────────────────────────────────────────┐
│  甲方：XX科技有限公司   乙方：YY外包公司                  │
│  金额：￥500,000       期限：2026-05-01 ~ 2026-12-31    │
└────────────────────────────────────────────────────────┘

┌── 风险项（3个）────────────────────────────────────────┐
│  🔴 高风险  第8条：乙方可单方面修改交付物定义            │
│  🟡 中风险  第12条：违约金上限仅为合同金额1%            │
│  🟡 中风险  第15条：争议管辖地为乙方所在地             │
└────────────────────────────────────────────────────────┘

┌── AI 建议 ──────────────────────────────────────────┐
│  建议修改第8条，明确交付物定义需双方书面确认...         │
└────────────────────────────────────────────────────┘

[✅ 批准继续]  [✏️ 添加修改意见]  [❌ 拒绝合同]
```

---

## 七、评估标准

- [ ] 低风险合同（评分 < 30）自动跳过人工审核直接通过
- [ ] 高风险合同正确触发 interrupt，前端显示审核界面
- [ ] 服务重启后，暂停中的审核任务仍可恢复
- [ ] 审核人批准后，Graph 从断点继续执行至完成
- [ ] 审计日志完整记录每次人工决策
- [ ] 多轮修改（AI 改 → 人工审核 → 再改）正常流转

---

## 八、项目交付物

1. `app.py` — Streamlit 前端
2. `graph/workflow.py` — LangGraph 工作流
3. `graph/nodes/` — 各节点实现
4. `graph/state.py` — 状态定义
5. `api/review.py` — 接收人工决策的 API 端点
6. `db/audit_log.py` — 审计日志
7. `README.md` — 完整部署文档

---

## 九、扩展方向

- 集成 OCR，支持扫描版 PDF 合同
- 引入合同模板库，与标准版本对比差异
- 支持移动端审核（微信/钉钉小程序）
- 多级审批流（法务 → 业务负责人 → CEO）
- 与 DocuSign 等电子签名平台集成
