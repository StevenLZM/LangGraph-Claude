# Production-Grade Agent Evals 教学文档

本文基于 `03_MULTI_AGENT` 的 InsightLoop 多 Agent 系统，讲清楚一个生产级 Agent eval 体系如何设计、建立、运行、解读和持续迭代。它不是单个测试脚本说明，而是一套从离线测试到真实链路门禁、从自动评分到人工复核闭环的完整教学材料。

## 0. 如何使用本文

这份文档可以按三种方式使用：

| 读者 | 推荐读法 | 目标 |
|---|---|---|
| 学生 / 新同学 | 先读 1-3 节理解为什么要分层，再按 12-14 节跑命令和解读结果 | 建立生产级 Agent eval 的完整心智模型 |
| 项目维护者 | 重点看 10-15 节，确认报告、gate、抽检和人工复核是否随代码同步 | 把 eval 变成发布流程的一部分 |
| 面试官 | 直接使用 16-17 节的问题、期望回答和评分参考 | 判断候选人是否能把 eval 落到工程闭环 |

阅读时要把 `evals/` 当成一个小型质量平台，而不是一个脚本目录。`evals.run` 负责执行和记录，`judge` 负责结果评分，`diagnostics` 和 `trace_metrics` 负责解释根因，`production` 负责发布门禁，`manual_review` 负责把无反馈场景下的质量判断变成可回流数据。

## 1. 教学目标

读完本文后，学生应该能回答三个问题：

- 为什么 Agent eval 不能只看最终答案分数。
- 如何把结果质量、过程链路质量、运行稳定性组织成一个可落地的评估体系。
- 在没有用户反馈的情况下，如何用风险抽检和人工复核数据持续改进 Agent。

本项目的目标不是展示“某个评测框架很高级”，而是展示生产系统真正需要的组合拳：

| 能力 | 解决的问题 | 本项目实现 |
|---|---|---|
| 离线单测 | 保护确定性契约 | `tests/` |
| 离线 graph E2E | 不依赖真实 key 验证图闭环 | `tests/test_end_to_end_offline.py` |
| Golden dataset | 固定题集做版本回归 | `evals/dataset.jsonl` |
| LLM-as-judge | 评估最终报告质量 | `evals/judge.py` |
| 确定性诊断 | 用代码解释扣分原因 | `evals/diagnostics.py` |
| Trace metrics | 记录真实链路调用、错误、token、耗时 | `evals/trace_metrics.py` |
| 生产门禁 | 用 release gate 决定是否放行 | `evals/production.py` |
| 人工复核 | 无用户反馈时建立质量标签闭环 | `evals/manual_review.py` |
| 看板和报告 | 让评测结果可读、可比较 | `evals/report.py`, `app/evals_ui.py` |

## 2. 为什么 Agent Evals 要分三层

普通 LLM 应用常见的评估方式是“输入问题，比较答案”。但 Agent 系统会调用工具、规划任务、并行检索、反思补查、写报告，还会受到网络、外部 API、工具降级链和 checkpoint 状态影响。只看最终答案会漏掉大量生产风险。

本项目采用三层评估：

| 层级 | 核心问题 | 典型指标 |
|---|---|---|
| 结果质量 | 最终报告能不能交付？ | `coverage`, `accuracy`, `citation`, `overall` |
| 过程链路质量 | 哪个 Agent 阶段导致问题？ | `process_quality`, `retrieval_metrics`, `component_quality` |
| 运行稳定性 | 系统是否稳定、可复现、可排查？ | `runtime_health`, `node_metrics`, `sampling` |

这三层的关系是：结果质量告诉你“好不好”，过程链路质量告诉你“为什么”，运行稳定性告诉你“能不能在生产里长期跑”。

## 3. 03 项目的评测架构

InsightLoop 的真实 graph 链路是：

```text
planner -> supervisor -> research_subgraph -> reflector -> writer
```

评测链路包在真实 graph 外面：

```text
dataset.jsonl
    |
    v
evals.run
    |
    +-- 调真实 LangGraph graph
    |       |
    |       +-- LangSmith trace, agent:<node> tags
    |       +-- EvalTraceCollector callback -> node_metrics
    |
    +-- judge_one -> score
    +-- diagnostics -> process/retrieval/component/runtime/sampling
    +-- framework_coverage -> 教学覆盖矩阵
    |
    v
results/{run_id}/
    |
    +-- results.jsonl
    +-- REPORT.md
    +-- manual_review_queue.jsonl
    +-- PRODUCTION_REPORT.md 生产 runner 才生成
```

这个设计有几个关键点：

- 评测层不修改业务 graph，只通过 `RunnableConfig`、callback 和输出状态做观测。
- 自动评分和确定性诊断分开，避免 LLM-as-judge 成为唯一真相。
- 离线 eval 和生产 eval 使用同一套记录 schema，方便横向比较。
- 人工复核不是临时看报告，而是有队列、有标签、有根因字段的结构化数据。

## 4. 数据集设计

`evals/dataset.jsonl` 是 golden dataset。它不是随机问题集合，而是生产回归基线。

当前数据集按四类平衡：

| 类别 | 作用 |
|---|---|
| 技术 | 检查框架、架构、技术细节类问题 |
| 产业 | 检查行业信息和趋势分析 |
| 对比 | 检查多对象比较、维度展开和结论组织 |
| 追问 | 检查上下文延展和深挖能力 |

设计 golden dataset 时要注意：

- Case 要稳定，不能依赖短期新闻。
- Case 要覆盖核心业务路径，而不是只覆盖容易答的问题。
- 每条 case 应包含 `id`, `category`, `query`, `audience`。
- 数据集不需要一开始很大，但要有代表性和可维护性。

一个生产团队通常会维护三种集合：

| 集合 | 来源 | 用途 |
|---|---|---|
| Golden set | 人工设计的核心问题 | 每次版本回归 |
| Regression set | 历史失败 case | 防止问题复发 |
| Review set | 抽检和人工复核沉淀 | 持续校准风险规则和 judge rubric |

## 5. LLM-as-Judge 的角色

`evals/judge.py` 负责最终报告评分。它输入：

- 原始研究问题。
- Planner 子问题。
- Evidence 摘要。
- 最终报告全文。

输出结构化分数：

| 字段 | 含义 |
|---|---|
| `coverage` | 是否覆盖问题和子问题 |
| `accuracy` | 事实和推理是否可靠 |
| `citation` | 引用是否充分、对应 evidence |
| `overall` | 综合可交付程度 |
| `rationale` | 扣分理由 |

生产里不能盲信 LLM-as-judge，原因有三点：

- Judge 本身会漂移，模型升级后标准可能变化。
- Judge 对工具失败、空 evidence、超时等运行问题感知不完整。
- Judge 给的是结果评价，不等于根因分析。

所以本项目把 LLM-as-judge 放在结果质量层，再用确定性诊断和 trace 指标补齐解释能力。

## 6. 确定性诊断

`evals/diagnostics.py` 负责把 graph 输出转成稳定、可解释的诊断字段。

### 6.1 `process_quality`

回答“过程有没有明显断点”：

| 指标 | 含义 |
|---|---|
| `plan_size` | Planner 拆了多少子问题 |
| `evidence_count` | 总 evidence 数 |
| `evidence_per_subq` | 每个子问题的证据覆盖 |
| `missing_subquestions` | 哪些子问题没有 evidence |
| `source_diversity` | 来源类型多样性 |
| `forced_completion` | Reflector 是否硬兜底完成 |
| `citation_audit_issue_count` | Writer 引用审计问题数 |

### 6.2 `retrieval_metrics`

回答“检索是否覆盖计划”：

| 指标 | 含义 |
|---|---|
| `subquestion_recall` | 有 evidence 的子问题占比 |
| `source_type_recall` | 覆盖了多少计划中的来源类型 |
| `evidence_density` | 平均每个子问题 evidence 数 |

这些不是严格信息检索学里的 recall，而是 Agent 过程中的 proxy metric。它们的价值是快速暴露 Research 阶段是否漏查。

### 6.3 `component_quality`

回答“哪个 Agent 可能出问题”：

| 组件 | 主要参考信号 |
|---|---|
| Planner | `plan_size` |
| Research | `subquestion_recall` |
| Reflector | `forced_completion`, missing subquestions |
| Writer | citation score, citation audit |
| Runtime | error, slow case |

这类评分不是为了替代真实调试，而是为了把排查方向从“整条链路有问题”收敛到“优先看 Research 或 Writer”。

## 7. Callback Trace 和 LangSmith

生产级 Agent eval 必须接入 trace。否则你只能知道 case 失败了，却很难知道失败发生在 LLM、工具、节点、网络还是外部 API。

本项目分两层做 trace：

| 层级 | 作用 | 实现 |
|---|---|---|
| 云端完整 trace | 查看完整调用树、输入输出和节点路径 | LangSmith |
| 本地汇总指标 | 进入 eval record 和生产 gate | `EvalTraceCollector` |

### 7.1 LangSmith 配置

本地 `.env` 配置：

```bash
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_sk_...
LANGSMITH_PROJECT=insightloop-multi-agent
```

兼容变量：

```bash
LANGCHAIN_API_KEY=lsv2_sk_...
LANGCHAIN_PROJECT=insightloop-multi-agent
```

不要把真实 key 写入仓库。`.env.example` 只保留占位符。

### 7.2 业务维度 metadata

`evals.run` 给每次 graph 调用注入：

| 字段 | 用途 |
|---|---|
| `eval_run_id` | 定位某次评测 |
| `case_id` | 定位单条 case |
| `category` | 按类别筛选 |
| `audience` | 按受众筛选 |
| `research_query` | 搜索原始问题 |
| `app` | 区分 eval 流量 |

Tags 包含：

```text
eval
run:<run_id>
case:<case_id>
agent:<node>
```

### 7.3 `node_metrics`

`EvalTraceCollector` 通过 LangChain callback 收集：

| 指标 | 含义 |
|---|---|
| `chain_runs`, `chain_errors` | Chain/节点运行数和错误数 |
| `llm_calls`, `llm_errors` | LLM 调用数和错误数 |
| `tool_calls`, `tool_errors` | 工具调用数和错误数 |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | token 用量 |
| `chain_elapsed_ms`, `llm_elapsed_ms`, `tool_elapsed_ms` | 分层耗时 |
| `by_agent` | 按 `agent:<node>` 聚合 |

本地 `node_metrics` 的意义是：即使不查询 LangSmith API，`results.jsonl` 和 `PRODUCTION_REPORT.md` 也能做门禁和趋势分析。

## 8. 无用户反馈下的抽检设计

生产中最常见的情况是：用户不会给每条答案打分。没有反馈不等于没有质量闭环。

本项目通过 `sampling` 字段做风险分层：

| 风险 | 触发条件示例 | 建议动作 |
|---|---|---|
| high | 执行失败、低分、零 evidence、引用问题、强制完成 | `manual_review` |
| medium | 中等分、部分子问题缺 evidence、慢 case | `spot_check` |
| low | 自动指标正常 | `random_sample` |

抽检策略的原则：

- 高风险必看，避免严重问题沉默进入生产。
- 中风险抽看，发现规则漏判或 judge 偏差。
- 低风险随机看，防止体系只覆盖已知问题。

## 9. 人工复核数据闭环

`manual_review_queue.jsonl` 是待复核队列，来源于 `sampling.review_action`。

示例：

```json
{"case_id":"tech_01","thread_id":"eval-...","risk_level":"high","review_action":"manual_review","reasons":["low_overall_score"],"suggested_label":"major_issue","status":"pending"}
```

人工复核最终写入 `manual_reviews.jsonl`：

```json
{"case_id":"tech_01","reviewer":"alice","label":"major_issue","notes":"引用不足","root_cause":"research"}
```

标签集合：

| 标签 | 含义 |
|---|---|
| `pass` | 可直接交付 |
| `minor_issue` | 小缺陷，不影响主要结论 |
| `major_issue` | 关键遗漏、错误引用、证据不足 |
| `unsafe_or_misleading` | 明显误导、编造、关键事实错 |
| `unjudgeable` | 信息不足，无法判断 |

人工复核数据的用途：

- 把高频失败 root cause 加入 regression set。
- 校准 `sampling` 风险规则。
- 校准 LLM-as-judge rubric。
- 调整生产 gate 阈值。
- 为后续 fine-tuning、prompt 优化或工具治理提供标注数据。

## 10. 报告和看板

每次 `evals.run` 生成：

| 文件 | 用途 |
|---|---|
| `results.jsonl` | 原始结构化记录，后续分析的事实来源 |
| `REPORT.md` | 静态评测报告 |
| `manual_review_queue.jsonl` | 人工抽检队列 |

生产 runner 额外生成：

| 文件 | 用途 |
|---|---|
| `PRODUCTION_REPORT.md` | 生产门禁结论和 gate 失败原因 |

`REPORT.md` 包含：

- 明细表。
- 维度均值。
- 三层评估覆盖。
- 过程链路质量。
- RAG/Retrieval Proxy。
- 组件质量评分。
- 运行稳定性。
- Trace Metrics。
- 抽检建议。
- 人工复核队列。
- 教学评估框架覆盖矩阵。
- 失分案例。

`app/evals_ui.py` 用于：

- 单 run 详情查看。
- 两 run 分数对比。
- 展示 process/retrieval/component/trace/sampling 字段。

## 11. 生产门禁

`evals.production` 会调用真实 graph、真实 LLM、真实工具链、真实 judge 和报告管线。它不是 mock 测试。

运行：

```bash
make eval-prod-smoke
make eval-prod
```

等价命令：

```bash
PYTHONPATH=. python -m evals.production --limit 1
PYTHONPATH=. python -m evals.production --limit 0
```

生产前置检查：

| 环境变量 | 作用 |
|---|---|
| `DEEPSEEK_API_KEY` | 真实 LLM |
| `TAVILY_API_KEY` / `DASHSCOPE_API_KEY` / `BRAVE_API_KEY` | 至少一个真实搜索 provider |

默认 gate：

| Gate | 默认值 | 目的 |
|---|---:|---|
| `min_success_rate` | `1.0` | 烟测不允许失败 |
| `min_avg_overall` | `80` | 平均交付质量 |
| `min_case_overall` | `70` | 单 case 底线 |
| `min_avg_subquestion_recall` | `0.7` | 检索覆盖底线 |
| `max_avg_elapsed_sec` | `600` | 性能底线 |
| `max_high_risk_cases` | `0` | 高风险 case 不放行 |
| `max_tool_error_rate` | `0.0` | 工具错误不静默 |
| `max_llm_errors` | `0` | LLM 错误不静默 |

生产门禁的价值不是“所有指标永远完美”，而是把发布风险显性化。团队可以按业务阶段调阈值，但不能没有阈值。

## 12. 建立这套体系的推荐顺序

如果从零建设，不要一开始就追求完整平台。推荐按以下顺序迭代：

1. 建立最小 golden dataset。
2. 加 LLM-as-judge，先得到结果质量分。
3. 加确定性诊断，解释扣分原因。
4. 加离线 graph E2E，保护核心流程。
5. 加 callback trace，记录 LLM/tool/token/耗时。
6. 加生产 runner，跑真实 graph 和真实工具链。
7. 加 release gate，把评测变成发布条件。
8. 加无反馈抽检和人工复核队列。
9. 加看板和 run-to-run 对比。
10. 定期把人工复核失败 case 沉淀回 regression set。

不要跳过第 3 步。只有 judge 分数没有诊断字段，团队很快会陷入“分数低但不知道改哪里”的状态。

### 12.1 第一版落地 PR 检查清单

如果把这套体系迁移到另一个 Agent 项目，第一版 PR 至少应能回答下面这些问题：

| 检查项 | 最小可接受答案 |
|---|---|
| 数据集从哪里来 | 有 5-20 条稳定 golden case，覆盖核心任务类型 |
| 如何运行 | 有 `make eval-smoke` 或等价命令，普通开发者能本地执行 |
| 输出在哪里 | 每次 run 有唯一目录，至少包含 `results.jsonl` 和可读报告 |
| 评分怎么来 | LLM-as-judge 有固定 rubric、结构化输出和 rationale |
| 如何定位根因 | 有确定性诊断字段，至少覆盖 plan、evidence、runtime error |
| 如何接入发布 | 有 smoke gate 或 production gate，失败原因可读 |
| 没有用户反馈怎么办 | 有风险抽检队列和人工复核标签模板 |
| 如何防止体系漂移 | eval 逻辑本身有单测，README/Makefile/看板同步更新 |

第一版不需要覆盖所有指标，但必须保证“跑得起来、看得懂、能阻断明显风险、能把失败沉淀回来”。

## 13. 使用流程

### 13.1 开发时

```bash
make test
make eval-smoke
```

看：

- 是否有单测失败。
- `REPORT.md` 是否出现明显低分或执行错误。
- `manual_review_queue.jsonl` 是否出现新高风险 case。

### 13.2 合并前

```bash
make eval
```

看：

- 平均 `overall` 是否达到目标。
- 是否有新增低分 case。
- `component_quality` 是否集中指向某个模块。
- 两次 run 对比是否有显著退化。

### 13.3 发布前

```bash
make eval-prod-smoke
```

必要时：

```bash
make eval-prod
```

看：

- `PRODUCTION_REPORT.md` 是否 PASS。
- LangSmith 中对应 `metadata.eval_run_id` 的 trace 是否完整。
- Tool/LLM error 是否为 0 或在可接受范围内。

### 13.4 人工复核后

处理 `manual_review_queue.jsonl`：

1. 打开对应 report。
2. 查看 `score.rationale`、`process_quality`、`node_metrics`。
3. 必要时去 LangSmith 看完整 trace。
4. 写入 `manual_reviews.jsonl`。
5. 把 `major_issue` 和 `unsafe_or_misleading` 加入 regression set 或修复 backlog。

## 14. 结果解读方法

看到低分时，不要只改 prompt。按以下顺序排查：

1. `error` 是否存在。
2. `runtime_health.error_type` 是什么。
3. `node_metrics.llm_errors` 或 `tool_errors` 是否异常。
4. `retrieval_metrics.subquestion_recall` 是否低。
5. `process_quality.missing_subquestions` 是哪些。
6. `citation_audit_issue_count` 是否大于 0。
7. `component_quality` 指向哪个 Agent。
8. `score.rationale` 是否和确定性诊断一致。
9. LangSmith trace 中具体失败节点和输入输出是什么。
10. 是否需要人工复核而不是立即改代码。

常见结论：

| 现象 | 更可能的问题 |
|---|---|
| `subquestion_recall` 低 | Researcher 或工具检索覆盖不足 |
| `citation` 低但 evidence 充足 | Writer 引用组织问题 |
| `forced_completion=true` | Reflector 补查失败或达到硬兜底 |
| `tool_error_rate` 高 | 工具 provider、网络、限流或参数问题 |
| `llm_errors` 高 | LLM provider、超时、限流或上下文过长 |
| judge 低分但诊断正常 | Judge rubric 可能需要校准，或 case 需要人工复核 |

## 15. 生产级成熟度检查清单

| 检查项 | 当前项目状态 |
|---|---|
| 有固定 eval dataset | 已覆盖 |
| 有 LLM-as-judge | 已覆盖 |
| 有确定性 guardrails | 已覆盖 |
| 有 RAG/Retrieval proxy | 已覆盖 |
| 有组件级归因 | 已覆盖 |
| 有真实 graph/LLM/tool 生产 runner | 已覆盖 |
| 有 LangSmith trace | 已覆盖 |
| 有本地 callback trace metrics | 已覆盖 |
| 有 release gate | 已覆盖 |
| 有无反馈抽检策略 | 已覆盖 |
| 有人工复核标签体系 | 已覆盖 |
| 有人工复核队列产物 | 已覆盖 |
| 有 run-to-run 对比看板 | 已覆盖 |
| 有离线 CI 友好测试 | 已覆盖 |
| 有 opt-in live smoke | 已覆盖 |

一个更大的生产系统还可以继续加：

- 线上采样真实用户请求。
- 影子流量 A/B。
- 成本预算 gate。
- 按用户分层的质量指标。
- 跨模型版本的 judge 校准集。
- 自动创建质量 issue 或工单。

### 15.1 常见反模式

| 反模式 | 为什么危险 | 更好的做法 |
|---|---|---|
| 只看 `overall` 总分 | 低分时不知道该修 Planner、Research 还是 Writer | 保留 process、retrieval、component、runtime 分层字段 |
| 只跑 mock，不跑 live smoke | 无法发现真实 LLM、工具、网络和限流问题 | 离线 CI 保稳定，发布前 opt-in 跑生产 smoke |
| 失败后重跑到通过 | 掩盖外部波动和模型随机性 | 记录失败频率，必要时进入 gate、重试策略或错误预算 |
| 人工复核只写一句评论 | 无法回流到数据集和规则 | 使用固定标签、root cause、notes 和 case id |
| 指标越多越好 | 团队不知道每个指标对应什么动作 | 每个指标都说明用途、阈值和后续处理 |
| eval 文档和命令不同步 | 新同学无法复现，发布流程失效 | Makefile、README、教学文档和测试一起更新 |

## 16. 面试问题与期望回答

下面的问题按面试官视角设计，用于判断候选人是否真正理解生产级 Agent eval，而不是只会说“用 LLM-as-judge 评估一下”。

### Q1：为什么 Agent 系统不能只用最终答案分做评估？

期望回答：

Agent 的失败可能来自规划、工具调用、检索覆盖、反思补查、写作引用、运行时超时或外部 API。最终答案分只能说明结果好坏，不能定位根因。生产级 eval 应该至少包含结果质量、过程链路质量和运行稳定性三层指标。

优秀回答会补充：

- LLM-as-judge 有漂移和误判风险。
- 需要确定性 guardrails 和 trace 指标辅助。
- 需要把评测结果转成可行动的修复方向。

### Q2：你会如何设计一个 Agent eval dataset？

期望回答：

我会先从核心业务场景抽样，建立小而稳定的 golden dataset，覆盖主要任务类型、难度和用户受众。每条 case 有稳定 id、category、query 和必要 metadata。后续把线上失败、人工复核发现的问题加入 regression set。

优秀回答会补充：

- 不追求一开始很大，先追求代表性和可维护性。
- 避免强依赖短期新闻的问题。
- 区分 golden set、regression set 和 review set。

### Q3：LLM-as-judge 在生产 eval 中有什么风险？如何控制？

期望回答：

风险包括 judge 模型漂移、评分标准不稳定、对运行时问题感知不足、可能偏好某种表达风格。控制方式包括结构化输出、固定 rubric、保留 rationale、用确定性指标交叉验证、抽样人工复核、维护 judge calibration set。

优秀回答会补充：

- Judge 适合评结果，不适合单独做 release gate。
- 关键发布要结合成功率、错误率、trace、检索覆盖和人工复核。

### Q4：如果没有用户反馈，你怎么做有效抽检？

期望回答：

用风险分层抽检。执行失败、低分、空 evidence、引用审计问题、强制完成、工具错误等高风险 case 必须人工复核；中风险 spot check；低风险做随机抽样，防止未知问题被规则漏掉。

优秀回答会补充：

- 抽检结果要结构化保存，而不是临时看一眼。
- 人工标签要回流到 regression set、risk rule 和 judge rubric。

### Q5：什么是生产级 eval gate？和普通 eval report 有什么不同？

期望回答：

Eval report 用于分析，production gate 用于发布决策。Gate 应该有明确阈值，比如成功率、平均分、最低单 case 分、检索覆盖、平均耗时、高风险 case 数、工具错误率和 LLM 错误数。Gate fail 要返回可行动失败原因。

优秀回答会补充：

- Gate 阈值可以随阶段调整，但不能没有。
- 生产 smoke 应调用真实 graph、真实 LLM、真实工具链。
- 普通 CI 要保留离线测试，避免没有 key 时不可运行。

### Q6：为什么要同时接 LangSmith 和本地 callback `node_metrics`？

期望回答：

LangSmith 适合查看完整 trace、调用树和节点输入输出；本地 `node_metrics` 适合进入 `results.jsonl`、报告和 gate，不依赖查询 LangSmith API。两者互补：一个用于深度排查，一个用于自动化门禁和趋势分析。

优秀回答会补充：

- 本地指标应该按 agent 聚合，支持定位 planner/research/writer 等节点。
- Trace metadata 应包含 `eval_run_id` 和 `case_id`，方便从报告跳回 trace。

### Q7：如何判断问题出在 Research 还是 Writer？

期望回答：

看 `retrieval_metrics` 和 `process_quality`。如果 `subquestion_recall` 低、`missing_subquestions` 多、evidence 少，优先怀疑 Research。若 evidence 充足但 citation 分低、`citation_audit_issue_count` 高，优先怀疑 Writer。

优秀回答会补充：

- 还要看 LangSmith trace 中工具调用是否失败。
- Reflector 的 `forced_completion` 可能说明补查没有解决覆盖缺口。

### Q8：如果生产 eval 偶发失败，你会怎么处理？

期望回答：

先区分确定性代码问题、外部依赖波动和模型随机性。查看 `runtime_health.error_type`、`node_metrics.tool_errors/llm_errors` 和 LangSmith trace。如果是外部 API 波动，要考虑重试、限流、fallback 和错误预算；如果是模型输出不稳定，要收敛 prompt、结构化输出或降低随机性。

优秀回答会补充：

- 不应简单重跑直到通过。
- 偶发失败也要记录，并按频率决定是否进入 gate。

### Q9：人工复核标签怎么设计才有用？

期望回答：

标签要少而稳定，能支持决策。比如 `pass`、`minor_issue`、`major_issue`、`unsafe_or_misleading`、`unjudgeable`。同时记录 reviewer、notes、root_cause，方便后续归因和回流。

优秀回答会补充：

- 标签要服务后续动作，而不是为了标注而标注。
- `major_issue` 和 `unsafe_or_misleading` 应进入 regression 或修复 backlog。

### Q10：如何让 eval 体系长期可维护？

期望回答：

保持 schema 稳定，报告可读，测试覆盖 eval 逻辑本身。区分离线 CI 和 live eval。每次新增指标都要说明用途、阈值和行动方式。定期清理无效 case，把人工复核结果回流。

优秀回答会补充：

- Eval 体系本身也需要单测。
- 不要把所有指标都塞进一个总分。
- 文档、Makefile、README 和看板要同步更新。

### Q11：为什么要有 component quality，而不是只看整体分？

期望回答：

整体分不能告诉团队该修哪个模块。Component quality 把 planner、research、reflector、writer、runtime 分开，能帮助定位问题并分配责任。

优秀回答会补充：

- Component score 是启发式诊断，不是绝对真理。
- 最终要结合 trace 和人工复核确认。

### Q12：你会如何把这套 eval 接入发布流程？

期望回答：

PR 阶段跑离线单测和 eval smoke；合并前跑完整离线 eval；发布前跑 production smoke 或全量 production eval。Gate fail 阻断发布，报告里给出失败原因。高风险 case 进入人工复核队列。

优秀回答会补充：

- Live eval 默认 opt-in，避免普通 CI 依赖外部 key。
- 对关键服务可以设置 nightly production eval。
- 发布后采样线上真实请求补充 review set。

### Q13：你怎么验证 eval 体系本身是可信的？

期望回答：

Eval 代码也要测试。至少要有 judge prompt 构造测试、诊断字段测试、报告渲染测试、production gate 阈值测试和人工复核队列测试。对于 live eval，要默认跳过，只在显式环境变量打开时调用真实 LLM 和工具。

优秀回答会补充：

- 要用构造样本覆盖低分、错误、慢 case、空 evidence、引用问题等边界。
- Gate failure message 要稳定，方便 CI 和发布系统读取。
- 文档中的命令和实际 Makefile 要由测试或审查保持一致。

### Q14：如果新增一个 eval 指标，你会怎么判断它该不该进生产 gate？

期望回答：

先判断这个指标是否稳定、可解释、和发布风险强相关，并且 gate fail 后是否有明确动作。如果指标只是分析用，可以进 report；如果能阻断明确风险，比如成功率、最低单 case 分、工具错误率、LLM 错误数，才适合进 production gate。

优秀回答会补充：

- 不要把 proxy metric 当成绝对真理。
- 新指标应先观察一段时间，再决定阈值。
- 阈值应按业务风险和阶段调整，但要记录调整原因。

### Q15：如果 judge 分数和人工复核结论冲突，你会怎么处理？

期望回答：

人工复核优先用于校准，不是简单覆盖 judge。先看冲突类型：是 judge rubric 不清、evidence 输入不足、报告被截断、人工标准不一致，还是 case 本身不可判。然后把结论回流到 judge rubric、calibration set、抽检规则或数据集。

优秀回答会补充：

- 不应只凭单个冲突就调整整体阈值。
- 高风险类别要保留人工复核记录和 root cause。
- 如果人工判断也不一致，需要统一标注准则。

### Q16：如何同时控制质量、成本和延迟？

期望回答：

质量指标不能脱离成本和延迟。生产 gate 除了分数，还应观察 `elapsed_sec`、tool/LLM 调用数、token、错误率和慢 case。优化时要区分“减少无效调用”和“牺牲必要检索”。例如可以限制补查轮次、缓存稳定来源、对低风险问题降级模型，但不能让核心 case 的 coverage 和 citation 明显下降。

优秀回答会补充：

- 成本优化也要用 eval 验证，不能只看账单。
- 不同用户层级或任务类型可以有不同 gate。
- Trace 是定位高成本节点的关键依据。

## 17. 面试评分参考

| 等级 | 表现 |
|---|---|
| 初级 | 只会说用测试集和 LLM-as-judge 打分 |
| 中级 | 能区分结果分、过程指标和运行指标 |
| 高级 | 能设计真实 graph/LLM/tool 的生产门禁和无反馈抽检闭环 |
| 资深 | 能把 eval 体系接入发布、观测、人工复核、数据回流和组织协作 |

面试官应重点观察候选人是否会把 eval 结果转成工程动作。不能落到修复、门禁、回归和数据闭环的 eval 设计，只是实验报告，不是生产体系。

## 18. 本项目对应文件索引

| 文件 | 作用 |
|---|---|
| `evals/dataset.jsonl` | Golden dataset |
| `evals/judge.py` | LLM-as-judge |
| `evals/diagnostics.py` | 过程、检索、组件、运行、抽检诊断 |
| `evals/trace_metrics.py` | Callback trace metrics |
| `evals/manual_review.py` | 人工复核队列和标签持久化 |
| `evals/frameworks.py` | 教学评估框架覆盖矩阵 |
| `evals/run.py` | 标准 eval runner |
| `evals/production.py` | 生产级真实链路 runner 和 gate |
| `evals/report.py` | Markdown 报告 |
| `app/evals_ui.py` | Streamlit 评测看板 |
| `tests/test_evals.py` | eval 诊断和报告测试 |
| `tests/test_trace_and_review.py` | trace 和人工复核测试 |
| `tests/test_production_eval.py` | 生产 gate 测试 |

## 19. 一句话总结

生产级 Agent eval 不是“跑几个问题看分数”，而是把真实链路、结果评分、过程诊断、trace 指标、发布门禁和人工复核数据闭环组合成一个持续运行的质量系统。
