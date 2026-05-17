# 01_RAG RAGAS 评估报告

- 用例数：5
- 评估框架：RAGAS

## RAGAS 指标

- Context Precision：平均 0.517，有效样本 5
- Context Recall：平均 0.400，有效样本 5
- Semantic Similarity：平均 0.714，有效样本 5

## 分类指标

- conceptual: cases=1, Context Precision=0.583, Context Recall=1.000, Semantic Similarity=0.921
- cross_section: cases=1, Context Precision=1.000, Context Recall=1.000, Semantic Similarity=0.821
- precise: cases=2, Context Precision=0.500, Context Recall=0.000, Semantic Similarity=0.530
- time: cases=1, Context Precision=0.000, Context Recall=0.000, Semantic Similarity=0.768

## 样本明细

- `concept_agent_core` [conceptual] Context Precision=0.583, Context Recall=1.000, Semantic Similarity=0.921
- `manual_langgraph` [precise] Context Precision=0.000, Context Recall=0.000, Semantic Similarity=0.362
- `invoice_latest` [time] Context Precision=0.000, Context Recall=0.000, Semantic Similarity=0.768
- `resume_precise` [precise] Context Precision=1.000, Context Recall=0.000, Semantic Similarity=0.697
- `hospital_plan` [cross_section] Context Precision=1.000, Context Recall=1.000, Semantic Similarity=0.821
