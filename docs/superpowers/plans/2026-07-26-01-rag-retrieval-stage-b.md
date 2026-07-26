# 01_RAG Retrieval Stage B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the approved seven-stage production retrieval path after Elasticsearch BM25/Dense/RRF by adding Child reranking, business features, Parent diversity, Token Budget assembly, one-pass generation, and evidence citations.

**Architecture:** Elasticsearch returns two real Child rankings and application Weighted RRF fuses them. A Cross-Encoder filters and ranks Child candidates, deterministic business features adjust the score, then SQLite hydrates Parents for exact/near deduplication, per-document quota, and MMR. A Token Budget assembler emits the exact final Parent context consumed by generation, while the existing production trace records each boundary without recomputation.

**Tech Stack:** Python 3.11+, Elasticsearch 8.19, LangChain, sentence-transformers Cross-Encoder, tiktoken, SQLite, pytest.

## Global Constraints

- Keep Child candidates through `business_fused_child`; hydrate Parent only after Child reranking.
- Stage sizes are BM25 50, Dense 50, RRF 80, Cross-Encoder 15, business fusion 15, diversified Parent 8, final Parent at most 6.
- Missing results are allowed; no stage pads or copies candidates to reach K.
- A configured Cross-Encoder threshold filters candidates; an empty result returns the evidence-insufficient response.
- Freshness contributes only to `latest` intent; historical/ordinary queries get no freshness bonus.
- Exact Parent dedup precedes near dedup, document quota, and MMR.
- Generation consumes exactly `final_context_parent` and uses stable `[S1]` evidence IDs.
- Do not call retrieval a second time during generation or evaluation.
- Do not invent qrels or publish an official Test report from empty datasets.

---

### Task 1: Stage-B Configuration

**Files:**

- Modify: `01_RAG/config.py`
- Modify: `01_RAG/.env.example`
- Test: `01_RAG/tests/test_rag_pipeline.py`

**Interfaces:**

- Produces `RerankConfig.SCORE_THRESHOLD`, `RAGConfig.RERANK_TOP_K`,
  `BUSINESS_FUSION_TOP_K`, `DIVERSIFIED_PARENT_TOP_K`,
  `FINAL_PARENT_TOP_K`, `MAX_PARENTS_PER_DOCUMENT`,
  `SIMILARITY_DEDUP_THRESHOLD`, `MMR_LAMBDA`,
  `FINAL_CONTEXT_TOKEN_BUDGET`, and configurable business weights.

- [ ] Write a test asserting exact approved K values and that all weights sum to 1.
- [ ] Run `pytest tests/test_rag_pipeline.py::TestConfig -q` and observe failure.
- [ ] Add environment-backed configuration with defaults: 15, 15, 8, 6, 2, 0.92, 0.70, and 6000 evidence tokens.
- [ ] Add `RERANK_SCORE_THRESHOLD=0.0` as the local calibration starting point and document that release evaluation must tune it on Dev.
- [ ] Run the focused test and commit.

### Task 2: Child Cross-Encoder and Business Fusion

**Files:**

- Modify: `01_RAG/rag/reranker.py`
- Create: `01_RAG/rag/business_fusion.py`
- Create: `01_RAG/tests/test_business_fusion.py`
- Modify: `01_RAG/tests/test_reranker.py`

**Interfaces:**

```python
def rerank_documents(
    query: str,
    docs: list[Document],
    *,
    enabled: bool | None = None,
    top_n: int | None = None,
    score_threshold: float | None = None,
    scorer: ScoreFunction | None = None,
) -> list[Document]: ...

def fuse_business_features(
    query: str,
    documents: Sequence[Document],
    *,
    time_intent: Mapping[str, Any] | None,
) -> list[Document]: ...
```

- [ ] Add a failing test proving a candidate below threshold is removed and the Top-15 ordering uses real `rerank_score`.
- [ ] Add failing business tests for the exact configurable formula, `latest` freshness, and no ordinary-query freshness bonus.
- [ ] Run the focused tests and observe failure.
- [ ] Implement min-max score normalization, clamped authority, latest-only freshness, normalized version rank, stable tie ordering, and `business_score` metadata.
- [ ] Run tests and commit.

### Task 3: Parent Aggregation, Deduplication, Quota, and MMR

**Files:**

- Create: `01_RAG/rag/postprocessor.py`
- Create: `01_RAG/tests/test_postprocessor.py`

**Interfaces:**

```python
def diversify_parent_candidates(
    child_documents: Sequence[Document],
    *,
    parent_docstore: ParentDocStore,
    top_k: int = 8,
    max_parents_per_document: int = 2,
    similarity_threshold: float = 0.92,
    mmr_lambda: float = 0.70,
) -> list[Document]: ...
```

- [ ] Add failing tests for best-Child score plus capped evidence bonus.
- [ ] Add failing tests for exact duplicate removal, near duplicate removal, and maximum two Parents per `doc_id`.
- [ ] Add a failing MMR test where a slightly lower relevance Parent from another document is selected over a near-identical high relevance Parent.
- [ ] Run the new tests and observe failure.
- [ ] Implement deterministic Parent hydration, SHA-256 exact content identity, vector cosine with token-shingle fallback, quota, and greedy MMR with stable tie breaks.
- [ ] Run tests and commit.

### Task 4: Token Budget Context Assembly

**Files:**

- Create: `01_RAG/rag/context_assembler.py`
- Create: `01_RAG/tests/test_context_assembler.py`

**Interfaces:**

```python
def assemble_final_context(
    documents: Sequence[Document],
    *,
    max_documents: int = 6,
    token_budget: int = 6000,
    tokenizer_name: str = "cl100k_base",
) -> list[Document]: ...
```

- [ ] Add failing tests that enforce maximum six Parents, no duplicate `parent_id`, and total evidence tokens within budget.
- [ ] Add a failing test that truncates an oversized first Parent at a paragraph or sentence boundary while preserving Metadata and adding `context_truncated=True`.
- [ ] Run tests and observe failure.
- [ ] Implement tiktoken counting, stable `[S1]` assignment, boundary truncation, and stop/skip behavior that never pads the result.
- [ ] Run tests and commit.

### Task 5: Seven Real Retrieval Stages

**Files:**

- Modify: `01_RAG/rag/retriever.py`
- Modify: `01_RAG/rag/elasticsearch_retrievers.py`
- Create: `01_RAG/tests/test_seven_stage_trace.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RetrievalPipelineComponents:
    bm25: Callable[[str, Mapping[str, Any]], list[Document]]
    dense: Callable[[str, Mapping[str, Any]], list[Document]]
    rrf: Callable[[list[Document], list[Document]], list[Document]]
    cross_encoder: Callable[[str, list[Document]], list[Document]]
    business_fusion: Callable[[str, list[Document], Mapping[str, Any]], list[Document]]
    diversify_parents: Callable[[list[Document]], list[Document]]
    assemble_context: Callable[[list[Document]], list[Document]]

@dataclass(frozen=True)
class RetrievalExecution:
    final_documents: tuple[Document, ...]
    trace: EvaluationTrace

def retrieve_with_trace(
    query: str,
    *,
    retrieval_context: Mapping[str, Any],
    components: RetrievalPipelineComponents | None = None,
) -> RetrievalExecution: ...
```

- [ ] Add the seven-stage Fake-component integration test from the production evaluation plan.
- [ ] Run it and observe failure.
- [ ] Orchestrate the seven production components, measure each call with `perf_counter`, and record the exact returned sequence once.
- [ ] Enforce shared auth/time Metadata Filter before both ES routes and reject a missing required trace for official evaluation.
- [ ] Run retrieval tests and commit.

### Task 6: One-Pass Generation and Citation Validation

**Files:**

- Modify: `01_RAG/rag/chain.py`
- Create: `01_RAG/tests/test_rag_execution_trace.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class RagExecution:
    answer: str
    source_documents: tuple[Document, ...]
    trace: EvaluationTrace
    generation_latency_ms: float

def generate_answer_from_documents(
    *,
    question: str,
    documents: Sequence[Document],
    chat_history: Sequence[Any],
) -> str: ...

def run_rag_with_trace(
    question: str,
    *,
    chat_history: Sequence[Any] = (),
    auth_context: dict[str, Any] | None = None,
) -> RagExecution: ...
```

- [ ] Add a failing test proving exactly one retrieval and generation from the same final tuple.
- [ ] Add failing tests for valid `[S1]`, one retry on a missing/unknown citation, and safe evidence-insufficient fallback after a second invalid answer.
- [ ] Run tests and observe failure.
- [ ] Extract generation from retrieval, format stable evidence IDs, validate citation IDs, and keep Streamlit response keys compatible.
- [ ] Run Chain tests and commit.

## Verification

- [ ] Run all Stage-B unit tests plus existing ES, reranker, chunking, Chain, and app tests.
- [ ] Run the isolated ES integration test with `RUN_ES_INTEGRATION=1`.
- [ ] Confirm `EvaluationTrace.missing_required_stages()` is empty for a real production retrieval after reviewed Dev data and real Embedding/Reranker availability exist.
- [ ] Continue with Tasks 8-11 of `2026-07-26-01-rag-production-evaluation.md`.
