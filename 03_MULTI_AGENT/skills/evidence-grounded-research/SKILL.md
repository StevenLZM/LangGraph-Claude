---
name: evidence-grounded-research
description: Use when an InsightLoop Reflector reviews research coverage from runtime-provided evidence quality metadata and coverage guardrails.
---

# Skill: evidence-grounded-research

把检索结果当作证据，而不是最终答案。

这个 skill 不负责让 LLM 自行发现或调用，也不要求 LLM 从零重新计算原始证据质量。runtime 已提供的 quality metadata（如 `support_level`、`confidence`、`extracted_claim`）和 coverage guardrails 是输入事实与保护栏；Reflector 的任务是在这些约束下审查覆盖度、缺口和补查方向。

## Rules

1. 优先使用 runtime 已提供的 quality metadata 和 `coverage_guardrails`，不要从零重新计算原始证据质量。
2. 每条 evidence 只能支持 snippet 或 `extracted_claim` 明确表达的 claim。
3. `direct` 表示证据直接回答当前子问题；`indirect` 表示需要少量推理；`background` 只提供背景；`irrelevant` 不应计入覆盖度。
4. 不得只凭模型常识把 `background` 或 `indirect` 升级成 `direct`。
5. `confidence` 反映该 evidence 对当前子问题的支撑强度，不代表事实绝对正确。
6. 证据不足时，明确列出缺口，不要补常识性结论。

## Output Discipline

- 优先保留来源、上下文和限制条件。
- 如果来源只支持局部结论，必须把 `limitations` 写清楚。
- 多条来源重复时，以支撑最直接、上下文最清楚的一条作为主证据。
- `coverage_by_subq` 可以综合判断，但不能无视 deterministic guardrails：只有背景证据时不要轻易给充分覆盖。
