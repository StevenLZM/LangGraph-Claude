# 05 RAGAS 评估体系设计

> 适用范围：`01_RAG` 离线评估体系  
> 当前状态：已放弃项目自定义 Recall/MRR/关键词规则分，统一使用 RAGAS

---

## 1. 设计目标

RAG 评估的目标不是给系统贴一个“好/坏”的标签，而是让每次改动都能被复盘：

- 改 chunk、parent-child、TopK、Dense/BM25 权重后，检索上下文是否更好
- 改 prompt、模型、上下文格式后，回答是否更忠实、更正确
- 时间类、跨页类、表格类问题是否被平均分掩盖
- 发布前 candidate 策略是否不低于 baseline

本项目此前有自定义评估体系：检索侧算 Recall、MRR、Parent Hit，生成侧算关键词覆盖和引用规则。该方案实现简单，但有三个问题：

- 指标偏工程规则，难判断“表达不同但语义正确”的答案
- 检索和生成分数口径不统一，难做端到端对比
- 自定义规则越来越多后维护成本上升，和社区评估方法脱节

因此当前版本改为 RAGAS-only：数据集只维护 `question` 和 `reference`，评估指标全部由 RAGAS 产生。

---

## 2. 总体架构

```text
evals/dataset.jsonl
  │
  │ load_dataset()
  ▼
case: id / category / query_type / question / reference
  │
  ├─ dry-run
  │    ├─ _call_dry_run_retrieval()
  │    ├─ _call_dry_run_generation()
  │    └─ _dry_run_ragas_evaluator()
  │
  └─ real-run
       ├─ rewrite_query(use_llm=False)
       ├─ retrieve_with_hybrid()
       ├─ create_chain_with_history()
       └─ ragas.evaluate()

  ▼
build_ragas_row()
  │
  ├─ user_input
  ├─ retrieved_contexts
  ├─ response
  └─ reference
  ▼
ragas_results.jsonl
summary.json
REPORT.md
```

核心文件：

| 文件 | 职责 |
|------|------|
| `evals/dataset.jsonl` | 人工 reference 评测集 |
| `evals/run.py` | 离线评估入口，串联检索、生成、RAGAS、报告 |
| `evals/ragas_adapter.py` | RAGAS 数据转换、指标选择、真实 evaluate 调用 |
| `evals/report.py` | 汇总 RAGAS 指标并生成 JSON/Markdown 报告 |
| `tests/test_retrieval_evals.py` | 覆盖 RAGAS schema、dry-run、报告、旧指标移除 |

---

## 3. 数据集设计

每条样本一行 JSON。最小字段：

```json
{
  "id": "manual_langgraph",
  "category": "precise",
  "query_type": "keyword",
  "question": "LangGraph 开发手册里提到的状态图是什么？",
  "reference": "LangGraph 的状态图用于把应用建模为节点和边组成的流程，节点处理状态，边决定执行路径。",
  "expected_sources": ["LangGraph_开发手册.pdf"]
}
```

字段说明：

| 字段 | 必填 | 用途 |
|------|------|------|
| `id` | 是 | 样本唯一标识，报告和回归对比用它定位问题 |
| `category` | 是 | 分组统计，例如 `conceptual`、`precise`、`time`、`cross_section` |
| `question` | 是 | 用户问题或人工构造的回归问题 |
| `reference` | 是 | RAGAS 对比用标准答案或标准事实陈述 |
| `query_type` | 否 | 更细的题型标签，便于人工分析 |
| `expected_sources` | 否 | 调试字段，用于人工排查来源，不参与自定义打分 |

`reference` 的写法要遵守三个原则：

- 写事实，不写评分规则
- 覆盖回答必须包含的关键信息
- 不把无关措辞写得过死，否则 `semantic_similarity` 和 `answer_correctness` 会受影响

时间类样本也用自然语言 reference 表达期望：

```json
{
  "id": "invoice_latest",
  "category": "time",
  "query_type": "time_latest",
  "question": "最新的发票是哪一张？",
  "reference": "知识库中的发票应按文档日期比较，选择日期最大的发票，并说明来源。",
  "expected_sources": ["珠海发票.pdf", "延庆发票.pdf", "新疆发票.pdf"]
}
```

---

## 4. RAGAS 输入映射

`build_ragas_row()` 将项目内部对象转换为 RAGAS 单轮评估字段：

| RAGAS 字段 | 来源 |
|------------|------|
| `user_input` | `case["question"]` |
| `retrieved_contexts` | 检索返回的 `Document.page_content` 列表 |
| `response` | RAG Chain 返回的 `answer` |
| `reference` | `case["reference"]` |

同时保留调试字段：

- `id`
- `category`
- `query_type`
- `retrieved_sources`
- `retrieved_parent_ids`
- `retrieved_sections`

这些字段不会送入 RAGAS 指标计算，但会写入 `ragas_results.jsonl`，用于人工复盘。

---

## 5. 指标选择

当前默认指标：

| 层级 | 指标 | 作用 |
|------|------|------|
| 检索质量 | `context_precision` | 检索上下文中有多少内容对回答有用 |
| 检索质量 | `context_recall` | reference 中的事实是否被上下文覆盖 |
| 语义质量 | `answer_relevancy` | 回答是否贴合用户问题 |
| 语义质量 | `semantic_similarity` | 回答与 reference 的语义接近程度 |
| 端到端质量 | `faithfulness` | 回答是否能被检索上下文支撑 |
| 端到端质量 | `answer_correctness` | 最终答案相对 reference 是否正确 |

读数时不要只看平均分。推荐顺序：

1. 先看 `context_recall`：如果需要的信息没召回，生成层很难补救。
2. 再看 `context_precision`：如果召回噪声太多，LLM 容易答散或引用错。
3. 再看 `faithfulness`：判断回答是否基于上下文，而不是模型补充。
4. 最后看 `answer_correctness`：判断端到端答案是否真正正确。

---

## 6. 运行模式

### 6.1 dry-run

```bash
python -m evals.run --dry-run
```

用途：

- 验证 JSONL 是否能加载
- 验证 `reference` 等必填字段
- 验证 RAGAS schema 转换
- 验证 `ragas_results.jsonl`、`summary.json`、`REPORT.md` 能生成

dry-run 不访问向量库、不调用 LLM、不代表真实质量分。报告中的分数来自固定 fixture，只用于检查管道。

### 6.2 真实 RAGAS 评估

```bash
python -m evals.run
```

真实模式会调用当前 RAG 链路：

```text
question
  → rewrite_query(use_llm=False)
  → retrieve_with_hybrid()
  → create_chain_with_history()
  → ragas.evaluate()
```

前置条件：

- 已安装 `ragas` 和 `datasets`
- 当前环境可创建项目 LLM
- 当前环境可创建项目 Embedding
- 向量库和 parent docstore 已有可检索内容

---

## 7. 输出设计

每次运行输出到：

```text
evals/results/<run_id>/
├── ragas_results.jsonl
├── summary.json
└── REPORT.md
```

文件说明：

| 文件 | 用途 |
|------|------|
| `ragas_results.jsonl` | 每条样本的输入、检索上下文来源和 RAGAS 指标 |
| `summary.json` | 聚合指标，可用于 CI 门槛或脚本对比 |
| `REPORT.md` | 面向人工复盘的 Markdown 报告 |

`summary.json` 的结构按全局和 category 聚合：

```json
{
  "total": 5,
  "metrics": {
    "context_precision": {"average": 0.82, "count": 5}
  },
  "by_category": {
    "time": {
      "total": 1,
      "metrics": {
        "answer_correctness": {"average": 0.76, "count": 1}
      }
    }
  }
}
```

---

## 8. 调参使用方法

推荐流程：

1. 跑 baseline：

   ```bash
   python -m evals.run
   ```

2. 保存 `summary.json` 和 `REPORT.md`。
3. 只改一个变量，例如：
   - `SEMANTIC_WEIGHT`
   - `SEMANTIC_TOP_K`
   - `BM25_TOP_K`
   - `FINAL_TOP_K`
   - `PARENT_TARGET_TOKENS`
   - `MAX_HYDRATED_PARENTS`
4. 重新跑 `python -m evals.run`。
5. 按全局、category、单样本三级对比。

常见现象：

| 现象 | 优先排查 |
|------|----------|
| `context_recall` 低 | chunk 边界、BM25 召回、TopK、query rewrite |
| `context_precision` 低 | 噪声文档、排序、rerank、Dense/BM25 权重 |
| `faithfulness` 低 | Prompt 约束、上下文格式、回答中混入模型常识 |
| `answer_correctness` 低 | 检索漏信息、reference 不清晰、生成链路没用对上下文 |
| 时间类样本低 | 日期 metadata、时间意图解析、过滤条件、日期排序 |

---

## 9. 生产落地原则

完整 RAGAS 评估应该离线做，不进入用户请求链路。

原因：

- 单个线上 query 通常没有 reference
- RAGAS 会额外调用评估 LLM 和 Embedding，延迟高、成本高
- 评估失败不应影响主链路可用性
- 发布判断需要固定样本集，而不是单次请求的即时分数

线上每次调用应记录低成本质量信号：

- `query`
- `rewritten_query`
- `time_intent`
- 命中的 `doc_id / parent_id / child_id`
- rank、score、来源、页码、章节
- 检索耗时、生成耗时、总耗时
- answer、引用来源、是否拒答
- 用户反馈：点赞、点踩、点击来源、追问

这些日志用于抽样沉淀新 case：

```text
线上日志 / 用户反馈
  → 选择高频、点踩、低分召回、错引来源问题
  → 人工编写或修订 reference
  → 加入 evals/dataset.jsonl
  → 下次 RAGAS 离线回归
```

---

## 10. 验收标准

- `python -m evals.run --dry-run` 能生成三类输出文件
- `python -m evals.run` 能调用 RAGAS 真实指标
- `tests/test_retrieval_evals.py` 覆盖：
  - dataset 必填字段
  - RAGAS row 构造
  - RAGAS 输出列名归一化
  - dry-run 报告生成
  - 旧自定义指标不再出现在报告中
- 文档中不再要求使用 `--with-generation`、`--with-judge`
- 文档中不再把自定义 Recall/MRR/关键词规则分作为当前项目评估输出

