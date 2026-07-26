# 01_RAG 生产级真实效果评测设计

## 1. 目标

建立一套只输出真实结果的 RAG 离线评测系统，用于回答：

- BM25、Dense、RRF、Cross-Encoder 等每一步分别带来了什么效果。
- 哪个阶段丢失了正确证据。
- 最终上下文是否完整、排序是否合理。
- LLM 回答是否正确且忠于真实上下文。
- Candidate 相比 Baseline 是否可以发布。

本文取代当前以 `Recall@5 + MRR + Hit@5 + RAGAS` 为主、包含 dry-run 固定分数的评测设计。

## 2. 核心原则

1. 评测调用真实 ES、Embedding、Reranker、生成 LLM 和 RAGAS Judge。
2. 产品评测不提供 dry-run，不生成固定 `1.0` 分数。
3. 单元测试可以使用 Mock 验证公式，但不能生成正式效果报告。
4. 评测直接采集生产检索链路，不重新实现一套检索逻辑。
5. 答案生成和 RAGAS 必须使用同一份最终上下文。
6. 任一必需阶段或外部依赖失败，本次运行失败。
7. 不合成单一总分，保留各阶段可诊断指标。

## 3. 总体架构

```text
train/dev/test 数据集
  ↓
真实 Query
  ↓
生产检索链路，同时记录 EvaluationTrace
  ├─ bm25_child@50
  ├─ dense_child@50
  ├─ rrf_child@80
  ├─ cross_encoder_child@15
  ├─ business_fused_child@15
  ├─ diversified_parent@8
  └─ final_context_parent@6
  ↓
使用 final_context 真实生成一次答案
  ↓
分阶段 IR 指标 + 真实 RAGAS
  ↓
逐 Case 结果、汇总和发布对比报告
```

评测不再次调用 RAG Chain。生产链路应提供一个可观测入口：

```python
answer, trace = run_rag_with_trace(query, auth_context)
```

`answer` 与 `trace.final_context` 必须来自同一次执行。

## 4. 七阶段 Trace

每个 `StageSnapshot` 记录：

```text
stage_name
candidate_level: child | parent
configured_k
actual_count
latency_ms
candidates[]
```

每个 Candidate 至少记录：

```text
rank
child_id / parent_id / doc_id
source / page_range / section
score / score_type
```

`matched_evidence_ids` 由离线 `QrelsResolver` 计算，不写入生产 Trace，避免生产链路依赖评测金标。

七个正式阶段：

| 阶段 | 粒度 | K |
|---|---|---:|
| `bm25_child` | Child | 50 |
| `dense_child` | Child | 50 |
| `rrf_child` | Child | 80 |
| `cross_encoder_child` | Child | 15 |
| `business_fused_child` | Child | 15 |
| `diversified_parent` | Parent | 8 |
| `final_context_parent` | Parent | 6 |

Query 标准化、实体识别、意图识别和权限 Filter 记录为 Trace Metadata，不作为排名阶段。

正式 Test 报告要求七个阶段全部存在。当前生产链路尚未实现的阶段不得用复制上一阶段结果或固定分数代替。

## 5. 数据集隔离

```text
01_RAG/evals/datasets/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── manifest.json
```

用途：

- `train`：训练或校准模型、阈值和业务权重。
- `dev`：调整 Top K、RRF、Rerank、MMR 等超参数。
- `test`：冻结的真实发布门禁。

采用约 `60% / 20% / 20%` 的分层分组划分：

- 相似问题只能进入一个集合。
- 同一业务事件的问题必须在同一集合。
- 尽量按来源文档族分组，避免证据泄漏。
- 各集合保持 Query 类型、时间、难度、文档类型和权限场景分布。

当前 5 条样本仅作为种子。Test 至少需要 200 条人工审核样本，完整数据集逐步达到约 1000 条。

## 6. Case 与 qrels

```json
{
  "id": "warranty_001",
  "category": "precise",
  "query_type": "keyword",
  "question": "产品保修期多久？",
  "reference": "产品保修期为 12 个月。",
  "expected_behavior": "answer",
  "auth_context": {
    "tenant_id": "tenant-1",
    "principals": ["role:employee"]
  },
  "qrels": [
    {
      "evidence_id": "warranty-policy",
      "source": "产品手册.pdf",
      "page_range": "3",
      "section": "保修政策",
      "evidence_text": "产品保修期为 12 个月",
      "evidence_hash": "...",
      "grade": 3
    }
  ]
}
```

`expected_behavior` 支持：

- `answer`：应基于证据回答。
- `abstain`：知识库没有足够证据，应拒答。
- `deny`：用户没有权限，应拒绝访问。

`answer` Case 计算分阶段 IR 与 RAGAS；`abstain` 计算拒答正确率；`deny` 使用受限证据 qrels 计算权限泄漏率和拒绝正确率。后两类不参与普通 Recall/NDCG 聚合。

相关等级：

| Grade | 含义 |
|---:|---|
| 3 | 直接且完整支持答案的核心证据 |
| 2 | 有效支持部分答案 |
| 1 | 相关背景，但不足以回答 |
| 0 | 不相关 |

最终 qrels 必须人工审核。LLM 只能辅助生成标注初稿。

## 7. 稳定证据锚点

qrels 不直接绑定生成后的 `child_id` 或 `parent_id`，因为修改 Chunk 策略后这些 ID 可能变化。

稳定锚点使用：

```text
source + page_range + section + evidence_text/hash
```

匹配规则必须确定性执行：

1. `source` 必须一致。
2. Candidate 页码范围必须覆盖或相交。
3. 规范化后的 `evidence_text` 必须被 Candidate 内容覆盖。
4. 不调用 LLM 判断检索候选是否命中 qrel。

标注时应把 `evidence_text` 写成一个较短的原子事实，避免它跨越多个 Child。

## 8. 分阶段 IR 指标

每个阶段统一计算：

- `Recall@K`
- `NDCG@K`
- `MRR@K`
- `Hit@K`
- Candidate 数量
- 覆盖的有效证据数量

### 8.1 Recall@K

以 `grade >= 2` 的证据锚点为相关：

```text
Recall@K =
Top-K 覆盖的相关证据锚点数
/
全部相关证据锚点数
```

同一个证据锚点被多个相似 Chunk 命中，只计算一次。

### 8.2 NDCG@K

Gain：

```text
gain = 2^grade - 1
```

折损：

```text
DCG@K = Σ gain_i / log2(rank_i + 1)
NDCG@K = DCG@K / IDCG@K
```

为避免重叠 Chunk 重复抬高 NDCG，同一证据锚点只在首次出现时产生 Gain，后续重复命中 Gain 为 0。

### 8.3 MRR@K 与 Hit@K

- `MRR@K`：Top-K 中第一个 `grade >= 2` 证据的倒数排名；未命中为 0。
- `Hit@K`：Top-K 至少覆盖一个 `grade >= 2` 证据时为 1，否则为 0。

### 8.4 输出名称

例如：

```text
bm25_child_recall_at_50
bm25_child_ndcg_at_50
rrf_child_recall_at_80
cross_encoder_child_ndcg_at_15
final_context_parent_recall_at_6
final_context_parent_ndcg_at_6
```

报告同时展示相邻阶段的变化，用于定位召回或排序损失。

### 8.5 每阶段主要指标

底层结果为每个阶段保留全部四项 IR 指标，但报告突出该阶段最有诊断价值的指标：

| 阶段 | 主要指标 | 诊断目标 |
|---|---|---|
| `bm25_child@50` | Recall@50 | 关键词证据是否召回 |
| `dense_child@50` | Recall@50 | 语义证据是否召回 |
| `rrf_child@80` | Recall@80、NDCG@80 | 融合是否增加召回且保持合理排序 |
| `cross_encoder_child@15` | NDCG@15、MRR@15 | Rerank 是否把正确证据提前 |
| `business_fused_child@15` | NDCG@15 | 时间、版本和权威性是否改善排序 |
| `diversified_parent@8` | Recall@8、NDCG@8 | 去重和 MMR 是否误删有效证据 |
| `final_context_parent@6` | Recall@6、NDCG@6、Hit@6 | 最终交给 LLM 的证据是否完整 |

Recall 更适合诊断召回阶段，NDCG/MRR 更适合诊断排序阶段，Hit 用作最低保障检查。

## 9. 生成质量

生成层继续使用真实 RAGAS：

- Context Precision
- Context Recall
- Faithfulness
- Answer Correctness
- Answer Relevancy
- Semantic Similarity

Judge 模型、Embedding 模型、Prompt 和 RAGAS 版本必须写入运行清单。Judge 温度固定为 0。

Faithfulness 表示回答是否受上下文支持，不等价于引用正确。严格引用指标在回答链路完成 `[Sx]` 引用校验后另行接入。

## 10. 真实运行

命令：

```bash
python -m evals.run --split dev --run-id candidate-v1
python -m evals.run --split test --run-id release-v1
```

不提供 `--dry-run`。

运行前检查：

- 数据集、manifest 和 qrels 合法。
- ES、索引和 Parent Store 可用。
- Embedding、Reranker、生成 LLM 和 RAGAS Judge 可用。
- 七个阶段已注册。
- 模型、Prompt、语料、索引和数据集版本可记录。

任一检查失败，评测不开始。

## 11. Checkpoint 与失败

每个 Case 成功后写入 checkpoint。中断后允许恢复，但必须满足：

- 数据集版本相同。
- 语料和索引版本相同。
- 检索参数、模型和 Prompt 指纹相同。

运行状态：

```text
running
failed
completed
```

只有 `completed` 运行可以生成正式 `summary.json` 和 `REPORT.md`。

失败运行保留错误和已完成 Case，便于排查，但不能用于发布判断。

## 12. 输出

```text
evals/results/<run_id>/
├── run_manifest.json
├── case_results.jsonl
├── stage_metrics.jsonl
├── ragas_results.jsonl
├── failed_cases.jsonl
├── summary.json
└── REPORT.md
```

`run_manifest.json` 至少记录：

- Git commit。
- dataset/corpus/index 版本。
- Chunk、Embedding、Reranker、LLM、Judge 和 Prompt 版本。
- 全部检索参数。
- 配置指纹。
- 开始/结束时间和运行状态。

汇总维度：

- 全局。
- split。
- category。
- query_type。
- expected_behavior。

报告包含每个指标的平均值、有效样本数和 95% bootstrap 置信区间。

## 13. Baseline 与发布门禁

Candidate 与 Baseline 必须使用：

- 同一个冻结 Test。
- 同一语料版本。
- 同一 Judge 配置。
- 逐 Case 配对比较。

默认门禁：

1. 所有 Test Case 完成，不能忽略失败样本。
2. 权限泄漏为 0。
3. `final_context Recall@6` 和 `NDCG@6` 相比 Baseline 的回退不超过 0.02。
4. Faithfulness 和 Answer Correctness 回退不超过 0.02。
5. `abstain/deny` 正确率不能回退。
6. P95 端到端延迟增长不超过 20%，除非明确批准。

中间阶段指标主要用于诊断，不单独阻止发布。最终上下文、生成质量、安全和延迟负责门禁。

发布摘要只突出：

```text
BM25 Recall@50
Dense Recall@50
RRF Recall@80
Cross-Encoder NDCG@15
Final Context Recall@6
Final Context NDCG@6
Faithfulness
Answer Correctness
```

完整分阶段结果仍保存在明细文件中。系统不把这些指标加权合成为一个“RAG 总分”。

## 14. 组件边界

建议组件：

```text
EvaluationTrace
  → 保存生产链路真实阶段快照

QrelsResolver
  → 确定性匹配 Candidate 与证据锚点

StageMetrics
  → Recall/NDCG/MRR/Hit

RealEvaluationRunner
  → 预检、执行、checkpoint、恢复

RagasEvaluator
  → 真实 RAGAS Judge

ReportBuilder
  → 汇总、置信区间、Baseline 对比
```

现有 `evals/run.py` 不再同时承担数据加载、检索、生成、RAGAS 和报告全部职责。

## 15. 测试策略

单元测试可以 Mock，但只验证：

- qrels schema。
- 稳定证据匹配。
- Recall/NDCG/MRR/Hit 公式。
- 重复证据只计一次。
- Trace 完整性。
- 配置指纹和 checkpoint 恢复。
- 失败运行不生成正式报告。

真实集成测试验证：

- 从真实 ES 召回七阶段 Trace。
- final_context 与生成上下文一致。
- 真实生成和真实 RAGAS 能完成。
- 报告记录真实模型及版本。

测试 Fixture 的分数不得写入 `evals/results`。

## 16. 实施顺序

1. 定义 Trace、qrels 和指标接口。
2. 让生产检索链路逐阶段输出 Trace。
3. 修复当前“检索一次、生成时再次检索”的上下文不一致。
4. 删除 dry-run 产品入口。
5. 拆分 train/dev/test 并迁移现有 5 条种子样本。
6. 接入真实 RAGAS、checkpoint 和报告。
7. 扩展人工评测集。
8. 七阶段齐全后启用正式 Test 发布门禁。

如果某个生产阶段尚未实现，只能先完成接口和单元测试；不能生成缺少该阶段的正式 Test 报告。

## 17. 验收标准

1. CLI 不再提供 dry-run。
2. 正式结果全部来自真实生产链路。
3. 一次 Case 只检索一次、生成一次。
4. 七个阶段均输出可排序 Candidate。
5. 每阶段都有 Recall/NDCG/MRR/Hit。
6. NDCG 使用人工审核的 0～3 qrels。
7. train/dev/test 物理隔离。
8. Test 至少 200 条后才能作为生产发布门禁。
9. 外部依赖或 Case 失败时不生成正式报告。
10. Candidate/Baseline 支持逐 Case 和置信区间对比。
