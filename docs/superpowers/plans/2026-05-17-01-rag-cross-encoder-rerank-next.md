# 01_RAG Cross-Encoder Rerank Next Iteration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional cross-encoder rerank stage to improve final context ordering after Milvus Lite migration is stable.

**Architecture:** Keep current retrieval as candidate generation: Milvus dense retrieval plus BM25 ensemble and parent hydration. Add reranking behind an environment flag so baseline retrieval remains available for comparison.

**Tech Stack:** Python, LangChain Documents, sentence-transformers or a configurable rerank provider, pytest, RAGAS evaluation.

---

### Task 1: Reranker Interface

**Files:**
- Create: `01_RAG/rag/reranker.py`
- Test: `01_RAG/tests/test_reranker.py`
- Modify: `01_RAG/config.py`

- [ ] Add `RERANK_ENABLED=false`, `RERANK_MODEL`, and `RERANK_TOP_N` config values.
- [ ] Define a small `rerank_documents(query: str, docs: list[Document]) -> list[Document]` API.
- [ ] Add tests for disabled mode preserving input order and enabled mode sorting by injected scores.

### Task 2: Retrieval Integration

**Files:**
- Modify: `01_RAG/rag/retriever.py`
- Modify: `01_RAG/rag/chain.py`
- Test: `01_RAG/tests/test_rag_pipeline.py`

- [ ] Apply rerank after candidate retrieval and before final context formatting.
- [ ] Preserve existing parent hydration metadata such as `matched_child_ids`, `best_child_score`, and source fields.
- [ ] Add tests proving rerank changes order only when enabled.

### Task 3: Evaluation

**Files:**
- Modify: `01_RAG/evals/run.py`
- Modify: `01_RAG/README.md`

- [ ] Add an evaluation run note comparing baseline Milvus retrieval versus rerank-enabled retrieval.
- [ ] Report RAGAS context precision, context recall, and semantic similarity for both modes.
