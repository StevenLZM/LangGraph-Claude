# 01_RAG RAGAS 评估报告

- 用例数：5
- 评估框架：RAGAS

## RAGAS 指标

- Context Precision：平均 nan，有效样本 5
- Context Recall：平均 0.400，有效样本 5
- Faithfulness：平均 nan，有效样本 5
- Answer Correctness：平均 nan，有效样本 5
- Answer Relevancy：平均 nan，有效样本 5
- Semantic Similarity：平均 0.760，有效样本 5

## 分类指标

- conceptual: cases=1, Context Precision=0.583, Context Recall=1.000, Faithfulness=nan, Answer Correctness=0.780, Answer Relevancy=0.931, Semantic Similarity=0.937
- cross_section: cases=1, Context Precision=1.000, Context Recall=1.000, Faithfulness=nan, Answer Correctness=nan, Answer Relevancy=nan, Semantic Similarity=0.775
- precise: cases=2, Context Precision=0.375, Context Recall=0.000, Faithfulness=nan, Answer Correctness=nan, Answer Relevancy=nan, Semantic Similarity=0.656
- time: cases=1, Context Precision=nan, Context Recall=0.000, Faithfulness=0.500, Answer Correctness=0.361, Answer Relevancy=nan, Semantic Similarity=0.777

## 样本明细

- `concept_agent_core` [conceptual] Context Precision=0.583, Context Recall=1.000, Faithfulness=nan, Answer Correctness=0.780, Answer Relevancy=0.931, Semantic Similarity=0.937
- `manual_langgraph` [precise] Context Precision=0.000, Context Recall=0.000, Faithfulness=1.000, Answer Correctness=0.155, Answer Relevancy=0.000, Semantic Similarity=0.619
- `invoice_latest` [time] Context Precision=nan, Context Recall=0.000, Faithfulness=0.500, Answer Correctness=0.361, Answer Relevancy=nan, Semantic Similarity=0.777
- `resume_precise` [precise] Context Precision=0.750, Context Recall=0.000, Faithfulness=nan, Answer Correctness=nan, Answer Relevancy=nan, Semantic Similarity=0.693
- `hospital_plan` [cross_section] Context Precision=1.000, Context Recall=1.000, Faithfulness=nan, Answer Correctness=nan, Answer Relevancy=nan, Semantic Similarity=0.775
