# InsightLoop Evals Teaching Guide

`03_MULTI_AGENT/evals` is a teaching-oriented evaluation stack for an Agent
research workflow. It does not reduce Agent quality to a single score. It shows
how different testing and evaluation frameworks answer different questions.

For the full production-grade design/build/use walkthrough and interviewer
question bank, read [PRODUCTION_EVALS_TEACHING.md](./PRODUCTION_EVALS_TEACHING.md).

## 三层评估

| Layer | Question | Current signal |
|---|---|---|
| Result quality | 最终报告能不能交付？ | `score.coverage / accuracy / citation / overall` |
| Process quality | 哪个 Agent 阶段导致质量问题？ | `process_quality`, `retrieval_metrics`, `component_quality` |
| Runtime stability | 系统是否稳定、可复现、可排查？ | `runtime_health`, `sampling`, LangSmith tags |

## Teaching Frameworks Covered

| Framework | Purpose | 03 project coverage |
|---|---|---|
| Pytest contract/unit tests | 保护 reducer、registry、runtime skill、writer citation 等确定性契约 | `tests/` |
| Offline graph E2E | 验证 planner interrupt 到 writer 的完整离线闭环 | `tests/test_end_to_end_offline.py` |
| Golden dataset regression | 固定题集做版本回归 | `evals/dataset.jsonl` + `make eval` |
| LLM-as-judge | 用模型评估最终报告 coverage / accuracy / citation | `evals/judge.py` |
| Deterministic guardrails | 用代码检查引用、空 evidence、force complete 等硬信号 | `evals/diagnostics.py` |
| RAG/Retrieval proxy | 用 subquestion recall、source type recall、evidence density 近似检索覆盖 | `retrieval_metrics` |
| Component quality | 拆开 Planner / Research / Reflector / Writer / Runtime 做归因 | `component_quality` |
| Runtime reliability | 观察 error、elapsed_sec、slow_case、error_type | `runtime_health` |
| No-feedback sampling | 无用户反馈时按风险分层抽检 | `sampling.risk_level / review_action / reasons` |
| Human review rubric | 给人工抽检统一标签，不让人工复核变成主观闲聊 | `frameworks.HUMAN_REVIEW_LABELS` |
| Trace observability | 说明 trace 如何补齐节点级耗时、工具失败、token/cost | LangSmith trace + callback `node_metrics` |
| Run-to-run / A-B comparison | 比较两次 run 的整体分和单 case delta | `app/evals_ui.py` 两 run 对比 |

## Output Fields

Each row in `evals/results/{run_id}/results.jsonl` can contain:

- `score`: LLM-as-judge result quality score.
- `process_quality`: deterministic process signals such as `plan_size`,
  `evidence_count`, missing subquestions, source diversity, reflection action,
  and citation audit issues.
- `retrieval_metrics`: RAG/Retrieval proxy metrics.
- `component_quality`: component-level heuristic scores.
- `runtime_health`: execution status, elapsed time, slow-case flag, and error type.
- `node_metrics`: callback trace summary including chain / LLM / tool calls,
  errors, elapsed time, token usage, and `by_agent` aggregation from
  `agent:<node>` tags.
- `sampling`: no-feedback review action and risk reasons.
- `human_review`: optional finalized reviewer label that can be joined back from
  `manual_reviews.jsonl`.
- `framework_coverage`: teaching matrix showing which eval frameworks are covered,
  partial, template-only, or report-level.

## Human Review Rubric

Use this rubric when `sampling.review_action` is `manual_review` or `spot_check`:

| Label | Meaning |
|---|---|
| `pass` | 可直接交付 |
| `minor_issue` | 小缺陷，不影响主要结论 |
| `major_issue` | 遗漏、错误引用、证据不足或关键论证弱 |
| `unsafe_or_misleading` | 明显误导、编造、关键事实错 |
| `unjudgeable` | 证据不足，无法判断 |

Each eval run writes `manual_review_queue.jsonl` beside `results.jsonl`. Reviewers
can append final labels to `manual_reviews.jsonl` with the same schema:

```json
{"case_id":"tech_01","reviewer":"alice","label":"major_issue","notes":"引用不足","root_cause":"research"}
```

Use the finalized review labels to update future datasets, regression cases, and
gate thresholds. This closes the loop when there is no direct user feedback.

## Commands

```bash
make eval-smoke
make eval
make eval-prod-smoke
make eval-prod
streamlit run app/evals_ui.py
```

Use `REPORT.md` for a static review and `app/evals_ui.py` for run-to-run
comparison. The eval runner attaches a local callback collector to the real
graph and writes `node_metrics` into every record. When LangSmith is configured,
the same run is also visible in LangSmith with `tags=["eval", "run:<id>",
"case:<id>"]` and metadata such as `eval_run_id`, `case_id`, `category`, and
`research_query`.

For LangSmith, put the key in local environment or `.env`; do not commit it:

```bash
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_sk_...
LANGSMITH_PROJECT=insightloop-multi-agent
```

`LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` are also supported for LangChain
compatibility.

## Production Live Eval

Use the production runner when you want the test to call the real graph, real
LLM, configured search tools, writer, judge, and report pipeline:

```bash
PYTHONPATH=. python -m evals.production --limit 1
PYTHONPATH=. python -m evals.production --limit 0
```

It checks production prerequisites before running:

- `DEEPSEEK_API_KEY`
- one live search provider: `TAVILY_API_KEY`, `DASHSCOPE_API_KEY`, or `BRAVE_API_KEY`

After the live eval finishes, it writes:

- `results.jsonl`
- `REPORT.md`
- `PRODUCTION_REPORT.md`
- `manual_review_queue.jsonl`

`PRODUCTION_REPORT.md` applies release-style gates:

- success rate
- average overall score
- minimum case score
- average subquestion recall
- average elapsed time
- high-risk case count
- tool error rate from callback trace metrics
- LLM error count from callback trace metrics

There is also an opt-in pytest live smoke test:

```bash
INSIGHTLOOP_RUN_LIVE_EVAL=1 pytest tests/test_production_eval.py::test_live_production_eval_smoke -q
```

This pytest intentionally skips by default so ordinary offline CI remains stable.
