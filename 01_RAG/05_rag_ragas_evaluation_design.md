# 生产级 RAG 真实评测指南

## 1. 原则

评测只走真实生产链路：Elasticsearch BM25、Elasticsearch Dense、应用层 RRF、Cross-Encoder、业务特征融合、父块多样化、Token Budget、生成模型与 RAGAS Judge。

- 产品评测不提供模拟分数。
- 每个 Case 只执行一次 RAG；检索指标和 RAGAS 使用同一次执行的结果。
- 任一阶段缺失或降级，当前 Case 失败。
- 失败运行保留 checkpoint，但不能生成正式报告。
- 不把多项指标拼成一个“RAG 总分”。

## 2. 数据集

```text
evals/datasets/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── manifest.json
```

- Train：构造规则、Prompt 和训练数据，只用于开发。
- Dev：调参数、选方案。
- Test：冻结发布门禁，不参与调参；至少 200 条人工复核 Case。
- 三个 split 会检查重复 ID 和规范化问题文本，防止数据泄漏。

每条 JSONL Case 的核心字段：

```json
{
  "id": "warranty-001",
  "category": "precise",
  "query_type": "keyword",
  "question": "产品保修多久？",
  "reference": "产品保修期为 12 个月。",
  "expected_behavior": "answer",
  "auth_context": {"tenant_id": "tenant-a", "principals": ["employee"]},
  "qrels": [
    {
      "evidence_id": "warranty-policy",
      "source": "manual.pdf",
      "page_range": "3",
      "section": "保修政策",
      "evidence_text": "产品保修期为 12 个月",
      "evidence_hash": "<规范化 evidence_text 的 SHA256>",
      "grade": 3
    }
  ]
}
```

`expected_behavior`：

- `answer`：应基于证据作答；计算七阶段 IR 与 RAGAS。
- `abstain`：知识库证据不足；计算拒答准确率。
- `deny`：当前身份无权访问；计算安全拒绝准确率与权限泄漏。

qrels 以 `source + page_range + section + evidence_text/hash` 作为稳定锚点，不绑定会随 Chunk 策略变化的 `child_id`。Grade 0～3，Grade 2 以上视为有效答案证据。LLM 可以辅助起草，但 qrels 和 reference 必须人工复核。

## 3. 七阶段指标

| 阶段 | K | 报告重点 |
|---|---:|---|
| BM25 Child | 50 | Recall@50 |
| Dense Child | 50 | Recall@50 |
| RRF Child | 80 | Recall@80、NDCG@80 |
| Cross-Encoder Child | 15 | NDCG@15、MRR@15 |
| Business Fused Child | 15 | NDCG@15 |
| Diversified Parent | 8 | Recall@8、NDCG@8 |
| Final Context Parent | 6 | Recall@6、NDCG@6、Hit@6 |

每个阶段实际都会保存 Recall/NDCG/MRR/Hit、候选数与覆盖证据数。同一证据被重叠 Chunk 多次命中只计算一次。

- Recall 诊断召回损失。
- NDCG/MRR 诊断排序质量。
- Hit 是最低命中保障。
- 最终上下文指标最接近生成模型真正看到的证据。

## 4. 生成与安全指标

`answer` Case 使用真实 RAGAS：

- Context Precision
- Context Recall
- Faithfulness
- Answer Correctness
- Answer Relevancy
- Semantic Similarity

`abstain` 和 `deny` 不进入普通 Recall/NDCG 聚合。`deny` 的受限 qrels 若出现在任一返回 Candidate 或答案中，即记为权限泄漏；发布要求为 0。

## 5. 运行

```bash
# 开发集真实评测
python -m evals.run --split dev --run-id candidate-v1

# 冻结 Test 对比 Baseline
python -m evals.run --split test --run-id release-v1 \
  --baseline-run evals/results/baseline-v1
```

启动前会检查：

1. split、manifest、qrels 与人工复核状态。
2. Elasticsearch 与读取别名。
3. Parent Store。
4. Embedding、Cross-Encoder、生成 LLM 和 RAGAS Judge。
5. 七个生产阶段。
6. dataset、corpus、index、Git 与配置指纹。

空 split、未复核数据、缺少正式索引或模型不可用都会直接失败，不会用 fixture 填充分数。

## 6. 输出与 checkpoint

```text
evals/results/<run_id>/
├── run_manifest.json
├── case_results.partial.jsonl
├── failed_cases.jsonl
├── case_results.jsonl
├── stage_metrics.jsonl
├── ragas_results.jsonl
├── summary.json
└── REPORT.md
```

每个成功 Case 都先 fsync 到 checkpoint。`--resume` 仅在数据、语料、索引、Git 与配置指纹一致时恢复。只有 `status=completed`、完成数与预期数一致、失败数为 0 且所有 answer Case 都有七阶段指标时，才能发布最后五个正式文件。

报告按全局、split、category、query_type 和 expected_behavior 聚合；每项指标都有平均值、有效样本数和 95% bootstrap CI。

## 7. 发布门禁

Candidate 与 Baseline 必须使用同一冻结数据集、同一语料和同一 Case：

- 所有 Test Case 完成，且 Test 至少 200 条。
- 权限泄漏为 0，不允许退化检索。
- Final Context Recall@6、NDCG@6 回退不超过 0.02。
- Faithfulness、Answer Correctness 回退不超过 0.02。
- Abstain/Deny Accuracy 不回退。
- 端到端 P95 延迟增长不超过 20%。

BM25、Dense、RRF、Cross-Encoder 等中间指标用于定位问题，不单独阻止发布；最终上下文、生成质量、安全和延迟负责门禁。

## 8. 推荐工作流

1. 从真实流量、点踩和故障中抽样问题。
2. 人工标注 reference、行为类型与稳定 qrels。
3. 先放 Train/Dev，禁止读取 Test 调参。
4. 运行 Dev，查看相邻阶段 Recall/NDCG 的变化。
5. 固定 Candidate 后运行 Test，与 Baseline 配对比较。
6. 发布后持续记录 trace、引用、拒答、延迟和用户反馈，补充下一轮数据。

测试中的 Fake 只验证公式、失败策略和文件契约，永远不能生成生产评测报告。
