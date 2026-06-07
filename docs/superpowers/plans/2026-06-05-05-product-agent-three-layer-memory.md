# 05 Product Agent Three-Layer Memory Implementation Plan

## Objective

Upgrade `05_PRODUCT_AGENT` from lightweight session/user memory into a production-oriented three-layer memory system with retrieval decision, summary memory, and semantic long-term memory.

## Completed Changes

- Added retrieval decision support in `agent/retrieval.py`.
- Inserted `retrieval_decision` into the LangGraph flow.
- Added `SummaryMemoryStore` with SQLite/Postgres implementations.
- Added `SemanticMemoryStore` with SQLite fallback and optional Milvus Lite backend.
- Wired `/chat` to load session summaries, run retrieval decision, and conditionally retrieve keyword/semantic memories.
- Extended postprocess handling to write summary memory and semantic long-term memory asynchronously.
- Extended admin APIs and `/health` with summary/semantic memory visibility.
- Added environment, compose, README, and tests for the new memory architecture.

## Verification

Run from `05_PRODUCT_AGENT`:

```bash
pytest tests -q
```

Expected result after merge:

```text
93 passed
```

## Follow-Up Work

- Validate real embedding provider behavior against the current `SemanticMemoryStore` contract.
- Validate Milvus remote cluster deployment and recall quality.
- Add a real RocketMQ consumer worker and DLQ/replay workflow for postprocess events.
- Run 24-hour stability and formal Docker/Locust load-test reports.
