# 05 Product Agent Three-Layer Memory Design

## Summary

`05_PRODUCT_AGENT` now uses a production-oriented three-layer memory architecture:

- Session context: recent messages and session metadata scoped by `session_id`.
- Summary memory: rolling conversation summaries scoped by `session_id`.
- Long-term memory: cross-session user facts scoped by `user_id`, with keyword/structured storage and semantic retrieval.

The main `/chat` path remains responsible for idempotency, session locking, rate limits, retrieval decision, context assembly, graph execution, LLM generation, persistence, metrics, and outbox publishing. Summary and semantic memory writes run through asynchronous postprocess handling so user-visible responses are not blocked.

## Key Design

- `agent/retrieval.py` introduces `RetrievalDecision`, deterministic rules, and optional LLM JSON decision parsing. LLM decision failures fall back to rules.
- `agent/graph.py` includes `context_loader -> retrieval_decision -> agent -> finalizer`; API-level retrieval remains the source of truth when already computed.
- `memory/summary.py` stores rolling summaries with version, covered turns, source event, and update time.
- `memory/semantic.py` provides `StructuredMemory`, SQLite fallback, and optional Milvus Lite backend through `SEMANTIC_MEMORY_BACKEND=milvus`.
- `messaging/handlers.py` updates summary memory and semantic long-term memory when handling `PostprocessRequested`.
- Admin APIs expose session summaries and semantic memories for inspection.

## Data Flow

`/chat` flow:

```text
idempotency -> session lock -> rate/token budget
  -> load session + summary
  -> retrieval decision
  -> conditional keyword/semantic memory retrieval
  -> LangGraph context_loader/retrieval_decision/agent/finalizer
  -> required LLM answer
  -> quality evaluation + session save + metrics + RocketMQ outbox
```

Postprocess flow:

```text
PostprocessRequested
  -> idempotency by postprocess_event_ids
  -> quality evaluation
  -> SummaryMemoryStore.save_summary
  -> SemanticMemoryStore.upsert_from_turn
  -> SessionStore metadata update
```

## Configuration

- `SUMMARY_MEMORY_DB`: SQLite summary memory path.
- `SEMANTIC_MEMORY_BACKEND`: `sqlite`, `milvus`, or disabled values.
- `SEMANTIC_MEMORY_DB`: SQLite semantic memory path.
- `MILVUS_MEMORY_URI`: Milvus Lite or remote Milvus URI.
- `MILVUS_MEMORY_COLLECTION`: Milvus collection name.
- `MEMORY_EMBEDDING_DIMENSION`: vector dimension used by the current offline embedding fallback.
- `RETRIEVAL_DECISION_MODE`: `rules` by default, `llm` for structured LLM decisions.

## Boundaries

- Retrieval decision decides what context to retrieve; it does not make business safety decisions.
- Refund confirmation, human transfer, complaints, and legal-risk handling remain rule guardrails in `agent/service.py`.
- The Milvus backend is available, but the current embedding implementation is deterministic and offline-friendly; production should validate a real embedding provider and remote Milvus cluster before claiming semantic recall quality.
- Deleting user memory must clear both keyword memories and semantic memories.

## Tests

Covered by:

- `tests/test_retrieval_decision.py`
- `tests/test_summary_memory.py`
- `tests/test_semantic_memory.py`
- `tests/test_retrieval_integration.py`
- `tests/test_rocketmq_messaging.py`
- `tests/test_storage_backends.py`
- `tests/test_health.py`
- `tests/test_deployment.py`

Latest verification after merge to `main`:

```bash
cd 05_PRODUCT_AGENT && pytest tests -q
# 93 passed
```
