# 01_RAG Production Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dry-run-capable evaluation path with a real-only, seven-stage RAG evaluation system that computes staged Recall/NDCG/MRR/Hit and end-to-end RAGAS metrics from the same production execution.

**Architecture:** The production retrieval path emits typed stage snapshots into one `EvaluationTrace`; answer generation consumes the exact `final_context_parent` snapshot. The evaluation runner validates human-reviewed qrels, executes one real RAG request per case, checkpoints results, computes deterministic IR metrics plus real RAGAS metrics, and only publishes reports for complete runs.

**Tech Stack:** Python 3.11+, LangChain, Elasticsearch 8.19, RAGAS, pytest, JSONL.

## Global Constraints

- Product evaluation has no `--dry-run` and never writes fixed quality scores.
- Mock/Fake evaluators are allowed only in unit tests and never write under `evals/results`.
- Official Test reports require all seven real stages.
- One case performs one retrieval execution and one answer generation.
- Answer generation and RAGAS use the same `final_context_parent` documents.
- qrels use human-reviewed grades `0..3` and stable evidence anchors.
- Train, Dev, and Test are physically separate JSONL files.
- Official Test release gating requires at least 200 reviewed cases.
- Missing dependencies, missing stages, or failed cases prevent official summary/report creation.
- Do not invent business-fusion or MMR results before those production stages exist.
- Do not modify or stage unrelated workspace files.

**Design reference:** `docs/superpowers/specs/2026-07-26-01-rag-production-evaluation-design.md`

---

## File Structure

Create:

- `01_RAG/rag/retrieval_trace.py`: production-owned trace models and recorder.
- `01_RAG/evals/models.py`: evaluation case, qrel, manifest, and result models.
- `01_RAG/evals/qrels.py`: deterministic stable-evidence matching.
- `01_RAG/evals/checkpoint.py`: run state, fingerprints, checkpoint, and resume.
- `01_RAG/evals/statistics.py`: paired bootstrap confidence intervals.
- `01_RAG/evals/datasets/manifest.json`: dataset metadata and counts.
- `01_RAG/evals/datasets/train.jsonl`: reviewed training/calibration cases.
- `01_RAG/evals/datasets/dev.jsonl`: reviewed tuning cases.
- `01_RAG/evals/datasets/test.jsonl`: frozen release cases.
- `01_RAG/evals/annotations/README.md`: exact human review workflow.
- `01_RAG/evals/annotations/legacy_seed_cases.jsonl`: recoverable copy of the five legacy cases awaiting qrel review.
- `01_RAG/tests/test_retrieval_trace.py`: trace contract tests.
- `01_RAG/tests/test_qrels.py`: evidence matching tests.
- `01_RAG/tests/test_stage_metrics.py`: staged IR formula tests.
- `01_RAG/tests/test_eval_checkpoint.py`: checkpoint/resume tests.
- `01_RAG/tests/test_real_eval_runner.py`: real-only runner contract tests.
- `01_RAG/tests/test_eval_statistics.py`: confidence interval tests.

Modify:

- `01_RAG/rag/retriever.py`: emit actual retrieval stage snapshots.
- `01_RAG/rag/reranker.py`: expose scored Child rerank results when the target pipeline is available.
- `01_RAG/rag/chain.py`: generate from the exact traced final context.
- `01_RAG/evals/retrieval_metrics.py`: staged evidence-aware metrics.
- `01_RAG/evals/ragas_adapter.py`: enforce real product evaluation.
- `01_RAG/evals/run.py`: split selection, preflight, real execution, checkpoint.
- `01_RAG/evals/report.py`: staged reports, baseline comparison, confidence intervals.
- `01_RAG/tests/test_retrieval_evals.py`: remove dry-run expectations and cover real-only behavior.
- `01_RAG/05_rag_ragas_evaluation_design.md`: point to the approved production spec.
- `01_RAG/LEARNING_GUIDE.md`: teach staged versus end-to-end evaluation.
- `01_RAG/README.md`: replace dry-run commands.

---

### Task 1: Production Retrieval Trace Contract

**Files:**

- Create: `01_RAG/rag/retrieval_trace.py`
- Test: `01_RAG/tests/test_retrieval_trace.py`

**Interfaces:**

- Produces:

```python
REQUIRED_EVAL_STAGES: tuple[str, ...]

@dataclass(frozen=True)
class CandidateSnapshot:
    rank: int
    level: Literal["child", "parent"]
    doc_id: str
    parent_id: str
    child_id: str | None
    source: str
    page_range: str
    section: str
    score: float | None
    score_type: str
    page_content: str

@dataclass(frozen=True)
class StageSnapshot:
    name: str
    candidate_level: Literal["child", "parent"]
    configured_k: int
    latency_ms: float
    candidates: tuple[CandidateSnapshot, ...]

@dataclass
class EvaluationTrace:
    trace_id: str
    original_query: str
    rewritten_query: str
    metadata: dict[str, Any]
    stages: dict[str, StageSnapshot]

class StageRecorder:
    def __init__(
        self,
        *,
        trace_id: str,
        original_query: str,
        rewritten_query: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...
    @property
    def trace(self) -> EvaluationTrace: ...
    def record(
        self,
        *,
        name: str,
        documents: Sequence[Document],
        configured_k: int,
        score_type: str,
        latency_ms: float,
    ) -> StageSnapshot: ...
```

- [ ] **Step 1: Write failing trace-model tests**

```python
def test_stage_recorder_preserves_rank_and_identity():
    recorder = StageRecorder(trace_id="trace-1", original_query="保修期")
    docs = [
        Document(
            page_content="保修期为 12 个月",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "child_id": "child-1",
                "source": "manual.pdf",
                "page_range": "3",
                "section_path": "保修政策",
                "rrf_score": 0.02,
            },
        )
    ]

    stage = recorder.record(
        name="rrf_child",
        documents=docs,
        configured_k=80,
        score_type="rrf_score",
        latency_ms=12.5,
    )

    assert stage.candidates[0].rank == 1
    assert stage.candidates[0].child_id == "child-1"
    assert stage.candidates[0].score == 0.02
    assert recorder.trace.stages["rrf_child"] == stage


def test_trace_rejects_duplicate_stage_name():
    recorder = StageRecorder(trace_id="trace-1", original_query="q")
    recorder.record(
        name="bm25_child",
        documents=[],
        configured_k=50,
        score_type="bm25_score",
        latency_ms=1.0,
    )

    with pytest.raises(ValueError, match="重复阶段"):
        recorder.record(
            name="bm25_child",
            documents=[],
            configured_k=50,
            score_type="bm25_score",
            latency_ms=1.0,
        )
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd 01_RAG
pytest tests/test_retrieval_trace.py -q
```

Expected: FAIL because `rag.retrieval_trace` does not exist.

- [ ] **Step 3: Implement immutable snapshots and recorder**

Use `time.perf_counter()` at call sites; the recorder only accepts measured latency. Validate:

- Rank starts at 1.
- Child stages require `child_id`.
- Parent stages require `parent_id`.
- Stage names are unique.
- Stored documents are copied into immutable snapshots.

- [ ] **Step 4: Add completeness test**

```python
def test_trace_lists_missing_required_stages():
    trace = EvaluationTrace(
        trace_id="trace-1",
        original_query="q",
        rewritten_query="q",
        metadata={},
        stages={},
    )

    assert trace.missing_required_stages() == list(REQUIRED_EVAL_STAGES)
```

`REQUIRED_EVAL_STAGES` must be exactly:

```python
(
    "bm25_child",
    "dense_child",
    "rrf_child",
    "cross_encoder_child",
    "business_fused_child",
    "diversified_parent",
    "final_context_parent",
)
```

- [ ] **Step 5: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_retrieval_trace.py -q
git add rag/retrieval_trace.py tests/test_retrieval_trace.py
git commit -m "01_RAG add retrieval evaluation trace contract"
```

---

### Task 2: Dataset and Qrel Models

**Files:**

- Create: `01_RAG/evals/models.py`
- Create: `01_RAG/evals/datasets/manifest.json`
- Create: `01_RAG/evals/datasets/train.jsonl`
- Create: `01_RAG/evals/datasets/dev.jsonl`
- Create: `01_RAG/evals/datasets/test.jsonl`
- Create: `01_RAG/evals/annotations/README.md`
- Create: `01_RAG/evals/annotations/legacy_seed_cases.jsonl`
- Modify: `01_RAG/tests/test_retrieval_evals.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class EvidenceQrel:
    evidence_id: str
    source: str
    page_range: str
    section: str
    evidence_text: str
    evidence_hash: str
    grade: int
    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self: ...

@dataclass(frozen=True)
class EvaluationCase:
    id: str
    split: Literal["train", "dev", "test"]
    category: str
    query_type: str
    question: str
    reference: str
    expected_behavior: Literal["answer", "abstain", "deny"]
    auth_context: dict[str, Any]
    qrels: tuple[EvidenceQrel, ...]
    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        split: str,
    ) -> Self: ...

@dataclass(frozen=True)
class DatasetManifest:
    version: str
    status: str
    corpus_version: str
    split_counts: dict[str, int]
    split_sha256: dict[str, str]

def load_split(dataset_dir: Path, split: str) -> list[EvaluationCase]: ...
def validate_dataset(dataset_dir: Path, *, release_mode: bool) -> DatasetManifest: ...
```

- [ ] **Step 1: Write failing schema tests**

```python
def test_qrel_requires_grade_between_zero_and_three():
    with pytest.raises(ValueError, match="grade"):
        EvidenceQrel.from_dict({
            "evidence_id": "e-1",
            "source": "manual.pdf",
            "page_range": "3",
            "section": "保修",
            "evidence_text": "保修期为 12 个月",
            "evidence_hash": "abc",
            "grade": 4,
        })


def test_case_requires_reviewed_qrels_for_answer_behavior():
    with pytest.raises(ValueError, match="qrels"):
        EvaluationCase.from_dict({
            "id": "case-1",
            "category": "precise",
            "query_type": "keyword",
            "question": "保修多久",
            "reference": "12 个月",
            "expected_behavior": "answer",
            "auth_context": {},
            "qrels": [],
        }, split="test")
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_retrieval_evals.py -q
```

Expected: FAIL because the new models are absent.

- [ ] **Step 3: Implement strict models and loaders**

Validation rules:

- `id`, `question`, and `reference` are non-empty.
- `expected_behavior` is one of `answer`, `abstain`, `deny`.
- `auth_context` is a dictionary; identity and permission values are passed to the real production request.
- Answer cases require at least one grade `>=2` qrel.
- `grade` is an integer from 0 to 3.
- `evidence_id` is unique within a case.
- Dataset Case IDs are unique across all three files.
- `split` comes from the physical filename, not JSON input.

- [ ] **Step 4: Add split-leakage tests**

Create normalized question fingerprints by Unicode normalization, whitespace collapse, and lowercase English text.

```python
def _write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def _valid_case(case_id: str, question: str) -> dict:
    return {
        "id": case_id,
        "category": "precise",
        "query_type": "keyword",
        "question": question,
        "reference": "产品保修期为 12 个月。",
        "expected_behavior": "answer",
        "auth_context": {},
        "qrels": [{
            "evidence_id": f"{case_id}-evidence",
            "source": "manual.pdf",
            "page_range": "3",
            "section": "保修政策",
            "evidence_text": "保修期为 12 个月",
            "evidence_hash": "pending-resolver-validation",
            "grade": 3,
        }],
    }


def test_validate_dataset_rejects_same_question_in_train_and_test(tmp_path):
    _write_cases(
        tmp_path / "train.jsonl",
        [_valid_case("train-1", "产品保修多久？")],
    )
    _write_cases(
        tmp_path / "test.jsonl",
        [_valid_case("test-1", " 产品保修多久? ")],
    )
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="跨 split"):
        validate_dataset(tmp_path, release_mode=False)
```

- [ ] **Step 5: Add release-size gate**

```python
def test_release_mode_requires_two_hundred_test_cases(tmp_path):
    _write_cases(
        tmp_path / "test.jsonl",
        [_valid_case(f"test-{index}", f"问题 {index}") for index in range(199)],
    )
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="至少 200"):
        validate_dataset(tmp_path, release_mode=True)
```

- [ ] **Step 6: Create dataset files without fake labels**

- Move no unreviewed legacy case directly into Dev/Test.
- Copy the existing five cases byte-for-byte into `evals/annotations/legacy_seed_cases.jsonl` before deleting the legacy path in Task 10.
- Initialize the three JSONL files as empty tracked files.
- Set manifest counts to zero and `status="annotation_required"`.
- Document the exact review process in `evals/annotations/README.md`:

```text
1. Select a real query.
2. Locate source/page/section in the indexed corpus.
3. Copy a short atomic evidence_text.
4. Assign grade 0..3.
5. A second reviewer confirms reference and qrels.
6. Place the approved case into exactly one split.
7. Recompute manifest counts and hashes.
```

The software can be complete before 200 labels exist, but release-mode evaluation must remain blocked.

- [ ] **Step 7: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_retrieval_evals.py -q
git add evals/models.py evals/datasets evals/annotations/README.md tests/test_retrieval_evals.py
git commit -m "01_RAG define reviewed evaluation datasets"
```

---

### Task 3: Deterministic Stable-Evidence Resolver

**Files:**

- Create: `01_RAG/evals/qrels.py`
- Test: `01_RAG/tests/test_qrels.py`

**Interfaces:**

- Consumes: `CandidateSnapshot`, `EvidenceQrel`.
- Produces:

```python
@dataclass(frozen=True)
class EvidenceMatch:
    evidence_id: str
    grade: int

def normalize_evidence_text(text: str) -> str: ...
def hash_evidence_text(text: str) -> str: ...
def page_ranges_overlap(left: str, right: str) -> bool: ...
def match_candidate(
    candidate: CandidateSnapshot,
    qrels: Sequence[EvidenceQrel],
) -> tuple[EvidenceMatch, ...]: ...
```

- [ ] **Step 1: Write failing normalization and range tests**

```python
def test_normalize_evidence_text_ignores_whitespace_and_common_punctuation():
    assert normalize_evidence_text("产品保修期：12 个月。") == "产品保修期12个月"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("3", "3", True), ("2-4", "4-5", True), ("1-2", "3", False)],
)
def test_page_ranges_overlap(left, right, expected):
    assert page_ranges_overlap(left, right) is expected
```

- [ ] **Step 2: Write failing candidate-match test**

```python
def test_match_candidate_requires_source_page_and_evidence_text():
    candidate = CandidateSnapshot(
        rank=1,
        level="child",
        doc_id="doc-1",
        parent_id="parent-1",
        child_id="child-1",
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        score=1.0,
        score_type="bm25_score",
        page_content="本产品的保修期为 12 个月，从购买日起计算。",
    )
    qrel = EvidenceQrel(
        evidence_id="warranty",
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        evidence_text="保修期为 12 个月",
        evidence_hash=hash_evidence_text("保修期为 12 个月"),
        grade=3,
    )

    assert match_candidate(candidate, [qrel]) == (
        EvidenceMatch(evidence_id="warranty", grade=3),
    )
```

Add separate negative tests for wrong source, non-overlapping page, and missing evidence text.

- [ ] **Step 3: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_qrels.py -q
```

Expected: FAIL because `evals.qrels` does not exist.

- [ ] **Step 4: Implement deterministic matching**

Rules:

- Normalize Unicode with NFKC.
- Remove whitespace and common punctuation.
- Require exact source equality after path basename normalization.
- Require page-range overlap.
- Require normalized `evidence_text` to be a substring of normalized Candidate content.
- Verify `evidence_hash` against normalized `evidence_text`; reject corrupted labels.
- Do not call Embedding or LLM.

- [ ] **Step 5: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_qrels.py -q
git add evals/qrels.py tests/test_qrels.py
git commit -m "01_RAG match retrieved chunks to stable qrels"
```

---

### Task 4: Evidence-Aware Stage Metrics

**Files:**

- Modify: `01_RAG/evals/retrieval_metrics.py`
- Create: `01_RAG/tests/test_stage_metrics.py`
- Modify: `01_RAG/tests/test_retrieval_evals.py`

**Interfaces:**

- Consumes: `StageSnapshot`, `Sequence[EvidenceQrel]`.
- Produces:

```python
@dataclass(frozen=True)
class StageMetricResult:
    stage_name: str
    configured_k: int
    actual_count: int
    relevant_evidence_count: int
    covered_evidence_count: int
    recall_at_k: float | None
    ndcg_at_k: float | None
    mrr_at_k: float | None
    hit_at_k: float | None

def compute_stage_metrics(
    stage: StageSnapshot,
    qrels: Sequence[EvidenceQrel],
    *,
    expected_behavior: Literal["answer", "abstain", "deny"] = "answer",
) -> StageMetricResult: ...
```

- [ ] **Step 1: Write failing Recall/MRR/Hit test**

Use two relevant anchors and a Top-3 list that covers one at rank 2:

```python
def _qrel(evidence_id: str, text: str, grade: int) -> EvidenceQrel:
    return EvidenceQrel(
        evidence_id=evidence_id,
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        evidence_text=text,
        evidence_hash=hash_evidence_text(text),
        grade=grade,
    )


def _candidate(rank: int, text: str) -> CandidateSnapshot:
    return CandidateSnapshot(
        rank=rank,
        level="child",
        doc_id="doc-1",
        parent_id=f"parent-{rank}",
        child_id=f"child-{rank}",
        source="manual.pdf",
        page_range="3",
        section="保修政策",
        score=1.0 / rank,
        score_type="test_score",
        page_content=text,
    )


def _stage(texts: list[str]) -> StageSnapshot:
    return StageSnapshot(
        name="bm25_child",
        candidate_level="child",
        configured_k=len(texts),
        latency_ms=1.0,
        candidates=tuple(
            _candidate(rank, text)
            for rank, text in enumerate(texts, start=1)
        ),
    )


def test_stage_metrics_compute_recall_mrr_and_hit():
    qrels = [
        _qrel("e1", "保修期为 12 个月", 3),
        _qrel("e2", "保修从购买日起计算", 2),
    ]
    stage = _stage(["无关内容", "保修期为 12 个月", "其他内容"])

    result = compute_stage_metrics(stage, qrels)

    assert result.recall_at_k == 0.5
    assert result.mrr_at_k == 0.5
    assert result.hit_at_k == 1.0
```

- [ ] **Step 2: Write failing evidence-aware NDCG test**

Use:

- `e1` grade 3.
- `e2` grade 2.
- rank 1 matches `e2`.
- rank 2 repeats `e2`.
- rank 3 matches `e1`.

Expected:

```text
DCG = 3/log2(2) + 0/log2(3) + 7/log2(4) = 6.5
IDCG = 7/log2(2) + 3/log2(3)
NDCG ≈ 0.731
```

```python
def test_ndcg_counts_repeated_evidence_only_once():
    qrels = [
        _qrel("e1", "核心证据", 3),
        _qrel("e2", "部分证据", 2),
    ]
    stage = _stage(["部分证据", "部分证据", "核心证据"])

    result = compute_stage_metrics(stage, qrels)
    assert result.ndcg_at_k == pytest.approx(0.731, abs=0.001)
```

- [ ] **Step 3: Add zero-relevance behavior**

For `abstain` and `deny` cases:

- Retrieval Recall/NDCG/MRR/Hit are `None`, not misleading zeros.
- `abstain` uses abstention accuracy.
- `deny` may carry restricted qrels; matching any restricted qrel increments permission leakage instead of Recall.

- [ ] **Step 4: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_stage_metrics.py tests/test_retrieval_evals.py -q
```

- [ ] **Step 5: Implement metrics**

Implementation requirements:

- Relevant Recall denominator is unique qrels with grade `>=2`.
- A qrel can be credited once per stage.
- MRR searches only `stage.candidates[:configured_k]`.
- Gain is `2**grade - 1`.
- IDCG sorts unique qrel grades descending.
- Clamp floating results to `[0,1]` after numerical calculation.
- When `expected_behavior != "answer"`, return `None` for the four IR scores; the runner computes abstain/deny metrics separately.
- Preserve the old `compute_retrieval_metrics()` only until all callers migrate; remove it in Task 10.

- [ ] **Step 6: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_stage_metrics.py tests/test_retrieval_evals.py -q
git add evals/retrieval_metrics.py tests/test_stage_metrics.py tests/test_retrieval_evals.py
git commit -m "01_RAG add staged Recall and NDCG metrics"
```

---

### Task 5: Checkpoint, Fingerprint, and Run State

**Files:**

- Create: `01_RAG/evals/checkpoint.py`
- Create: `01_RAG/tests/test_eval_checkpoint.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class RunFingerprint:
    dataset_sha256: str
    corpus_version: str
    index_version: str
    git_commit: str
    config_sha256: str

class EvaluationCheckpoint:
    @classmethod
    def create(cls, run_dir: Path, fingerprint: RunFingerprint) -> Self: ...
    @classmethod
    def resume(cls, run_dir: Path, fingerprint: RunFingerprint) -> Self: ...
    def append_case(self, result: dict[str, Any]) -> None: ...
    def append_failure(self, failure: dict[str, Any]) -> None: ...
    def mark_failed(self) -> None: ...
    def mark_completed(self) -> None: ...
```

- [ ] **Step 1: Write failing fingerprint test**

```python
def _fingerprint(config_sha256: str) -> RunFingerprint:
    return RunFingerprint(
        dataset_sha256="dataset-v1",
        corpus_version="corpus-v1",
        index_version="index-v1",
        git_commit="deadbeef",
        config_sha256=config_sha256,
    )


def test_resume_rejects_changed_configuration(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.append_case({"id": "case-1"})

    with pytest.raises(ValueError, match="指纹"):
        EvaluationCheckpoint.resume(tmp_path, _fingerprint("config-b"))
```

- [ ] **Step 2: Write failing report-safety test**

```python
def test_failed_run_cannot_publish_summary(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.mark_failed()

    assert checkpoint.can_publish is False
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "REPORT.md").exists()
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_eval_checkpoint.py -q
```

- [ ] **Step 4: Implement atomic JSONL append and status**

Files:

```text
run_manifest.json
case_results.partial.jsonl
failed_cases.jsonl
```

Use explicit flush plus `os.fsync()` after each completed Case. `mark_completed()` only changes status; ReportBuilder publishes final files later.

- [ ] **Step 5: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_eval_checkpoint.py -q
git add evals/checkpoint.py tests/test_eval_checkpoint.py
git commit -m "01_RAG checkpoint real evaluation runs"
```

---

### Task 6: Retrieval Pipeline Trace Integration

**Dependency Gate:** Execute this task only after the Elasticsearch hybrid retrieval path and the real `business_fused_child`, `diversified_parent`, and `final_context_parent` stages exist. Until then Tasks 1–5 may merge, but official Test evaluation remains disabled.

**Files:**

- Modify: `01_RAG/rag/retriever.py`
- Modify: `01_RAG/rag/reranker.py`
- Modify target files created by the ES/post-processing implementation:
  - `01_RAG/rag/elasticsearch_retrievers.py`
  - `01_RAG/rag/postprocessor.py`
- Create: `01_RAG/tests/test_seven_stage_trace.py`

**Interfaces:**

- Consumes: `StageRecorder`.
- Produces:

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

- [ ] **Step 1: Write failing seven-stage integration test**

Inject deterministic Fake retrievers/rankers, but call the real orchestration function:

```python
def _child(child_id: str) -> Document:
    return Document(
        page_content=f"child {child_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": f"parent-{child_id}",
            "child_id": child_id,
            "source": "manual.pdf",
            "page_range": "3",
        },
    )


def _parent(parent_id: str) -> Document:
    return Document(
        page_content=f"parent {parent_id}",
        metadata={
            "doc_id": "doc-1",
            "parent_id": parent_id,
            "source": "manual.pdf",
            "page_range": "3",
        },
    )


def _complete_fake_components() -> RetrievalPipelineComponents:
    bm25_docs = [_child("bm25")]
    dense_docs = [_child("dense")]
    rrf_docs = [_child("rrf")]
    reranked_docs = [_child("reranked")]
    business_docs = [_child("business")]
    diversified_docs = [_parent("diverse")]
    final_docs = [_parent("final")]
    return RetrievalPipelineComponents(
        bm25=lambda query, context: bm25_docs,
        dense=lambda query, context: dense_docs,
        rrf=lambda bm25, dense: rrf_docs,
        cross_encoder=lambda query, docs: reranked_docs,
        business_fusion=lambda query, docs, context: business_docs,
        diversify_parents=lambda docs: diversified_docs,
        assemble_context=lambda docs: final_docs,
    )


def test_retrieve_with_trace_records_seven_real_stage_outputs():
    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"visibility": "public"},
        components=_complete_fake_components(),
    )

    assert tuple(execution.trace.stages) == REQUIRED_EVAL_STAGES
    assert execution.trace.stages["bm25_child"].configured_k == 50
    assert execution.trace.stages["rrf_child"].configured_k == 80
    assert execution.trace.stages["final_context_parent"].configured_k == 6
    assert [doc.metadata["parent_id"] for doc in execution.final_documents] == ["final"]
```

- [ ] **Step 2: Add identity-level assertions**

```python
assert all(c.level == "child" for c in trace.stages["cross_encoder_child"].candidates)
assert all(c.level == "parent" for c in trace.stages["diversified_parent"].candidates)
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_seven_stage_trace.py -q
```

- [ ] **Step 4: Record each actual stage**

At each production boundary:

1. Invoke the production component.
2. Measure elapsed time.
3. Attach the component’s real score to Document Metadata.
4. Call `StageRecorder.record()`.
5. Pass the same ordered Documents to the next component.

Do not recompute a stage for evaluation.

- [ ] **Step 5: Enforce actual K and identity**

- BM25: Child Top 50.
- Dense: Child Top 50.
- RRF: Child Top 80.
- Cross-Encoder: Child Top 15.
- Business fusion: Child Top 15.
- Diversity: Parent Top 8.
- Token Budget/final context: Parent maximum 6.

If Token Budget returns fewer than 6, store the actual count; do not pad.

- [ ] **Step 6: Add missing-stage failure test**

```python
def test_official_trace_rejects_missing_business_stage():
    execution = retrieve_with_trace(
        "保修期",
        retrieval_context={"visibility": "public"},
        components=_complete_fake_components(),
    )
    trace = execution.trace
    del trace.stages["business_fused_child"]

    with pytest.raises(ValueError, match="business_fused_child"):
        trace.require_complete()
```

- [ ] **Step 7: Run focused retrieval tests and commit**

```bash
cd 01_RAG
pytest tests/test_seven_stage_trace.py tests/test_chunking_v2.py tests/test_reranker.py -q
git add rag/retriever.py rag/reranker.py rag/elasticsearch_retrievers.py rag/postprocessor.py tests/test_seven_stage_trace.py
git commit -m "01_RAG trace seven production retrieval stages"
```

Only add target files that exist and were changed by this task.

---

### Task 7: One-Pass Retrieval and Generation

**Files:**

- Modify: `01_RAG/rag/chain.py`
- Create: `01_RAG/tests/test_rag_execution_trace.py`

**Interfaces:**

- Consumes: `retrieve_with_trace()`.
- Produces:

```python
@dataclass(frozen=True)
class RagExecution:
    answer: str
    source_documents: tuple[Document, ...]
    trace: EvaluationTrace
    generation_latency_ms: float

def run_rag_with_trace(
    question: str,
    *,
    chat_history: Sequence[Any] = (),
    auth_context: dict[str, Any] | None = None,
) -> RagExecution: ...
```

- [ ] **Step 1: Write failing one-retrieval test**

```python
def test_run_rag_with_trace_retrieves_once_and_generates_from_final_context(monkeypatch):
    import rag.chain as chain

    final_docs = [
        Document(
            page_content="产品保修期为 12 个月。",
            metadata={
                "doc_id": "doc-1",
                "parent_id": "parent-1",
                "source": "manual.pdf",
                "page_range": "3",
            },
        )
    ]
    trace = EvaluationTrace(
        trace_id="trace-1",
        original_query="保修期多久？",
        rewritten_query="产品保修期多久？",
        metadata={},
        stages={},
    )
    calls = {"retrieve": 0}

    def fake_retrieve(*args, **kwargs):
        calls["retrieve"] += 1
        return RetrievalExecution(
            final_documents=tuple(final_docs),
            trace=trace,
        )

    captured = {}

    def fake_generate(*, question, documents, chat_history):
        captured["documents"] = list(documents)
        return "保修期为 12 个月。"

    monkeypatch.setattr(chain, "retrieve_with_trace", fake_retrieve)
    monkeypatch.setattr(chain, "generate_answer_from_documents", fake_generate)

    execution = chain.run_rag_with_trace("保修期多久？")

    assert calls["retrieve"] == 1
    assert captured["documents"] == final_docs
    assert execution.source_documents == tuple(final_docs)
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd 01_RAG
pytest tests/test_rag_execution_trace.py -q
```

- [ ] **Step 3: Separate generation from retrieval**

Extract a function that accepts already-retrieved Documents:

```python
def generate_answer_from_documents(
    *,
    question: str,
    documents: Sequence[Document],
    chat_history: Sequence[Any],
) -> str: ...
```

`run_rag_with_trace()` calls `retrieve_with_trace()` once, then calls this generator with `execution.final_documents`.

- [ ] **Step 4: Keep Streamlit compatibility**

`create_rag_chain()` may wrap `run_rag_with_trace()` for UI compatibility, but must not start a second retrieval. Existing response keys remain:

```python
{"answer": answer, "sources": list(source_documents)}
```

- [ ] **Step 5: Run Chain tests and commit**

```bash
cd 01_RAG
pytest tests/test_rag_execution_trace.py tests/test_rag_pipeline.py tests/test_app_startup.py -q
git add rag/chain.py tests/test_rag_execution_trace.py
git commit -m "01_RAG generate answers from traced final context"
```

---

### Task 8: Real-Only Evaluation Runner

**Files:**

- Modify: `01_RAG/evals/run.py`
- Modify: `01_RAG/evals/ragas_adapter.py`
- Create: `01_RAG/tests/test_real_eval_runner.py`
- Modify: `01_RAG/tests/test_retrieval_evals.py`

**Interfaces:**

- Consumes: `load_split()`, `run_rag_with_trace()`, `compute_stage_metrics()`, `EvaluationCheckpoint`.
- Produces:

```python
def preflight_real_evaluation(
    *,
    split: str,
    dataset_dir: Path,
    release_mode: bool,
) -> RunFingerprint: ...

def run_real_evaluation(
    *,
    split: str,
    dataset_dir: Path,
    output_root: Path,
    run_id: str,
    baseline_run: Path | None = None,
    resume: bool = False,
) -> Path: ...

def build_parser() -> argparse.ArgumentParser: ...
```

- [ ] **Step 1: Remove dry-run product tests**

Delete tests that expect:

- `--dry-run`.
- `_call_dry_run_retrieval()`.
- `_call_dry_run_generation()`.
- `_dry_run_ragas_evaluator()`.
- `ragas_mode="dry_run_fixture"`.

Do not remove formula-level Fake tests.

- [ ] **Step 2: Write failing CLI test**

```python
def test_cli_has_no_dry_run_option():
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--dry-run" not in option_strings
    assert "--split" in option_strings
```

`--split` is required and limited to `train`, `dev`, `test`.

- [ ] **Step 3: Write failing preflight tests**

```python
def test_preflight_rejects_missing_required_stage(monkeypatch, valid_dataset):
    monkeypatch.setattr(run, "probe_trace_stage_names", lambda: {"bm25_child"})

    with pytest.raises(RuntimeError, match="缺少阶段"):
        preflight_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            release_mode=False,
        )


def test_preflight_rejects_unavailable_judge(monkeypatch, valid_dataset):
    monkeypatch.setattr(run, "probe_ragas_judge", lambda: False)

    with pytest.raises(RuntimeError, match="Judge"):
        preflight_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            release_mode=False,
        )
```

Preflight checks real ES/index/Parent Store/Embedding/Reranker/generation LLM/Judge availability through small, explicit probe functions.

The `valid_dataset` fixture creates physical `train.jsonl`, `dev.jsonl`, and `test.jsonl` files plus a matching manifest, with one reviewed Dev `answer` Case. Probe tests monkeypatch all unrelated probes to return success so each test isolates one failure.

- [ ] **Step 4: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_real_eval_runner.py tests/test_retrieval_evals.py -q
```

- [ ] **Step 5: Implement one-case real flow**

For each Case:

```text
RagExecution = run_rag_with_trace(question, auth_context=case.auth_context)
trace.require_complete()
resolve qrels for every stage
compute four IR metrics for every stage
if expected_behavior=answer:
    build RAGAS row from final_context + answer
    run real RAGAS
else:
    compute abstain/deny behavior metrics
append checkpoint
```

Do not call retrieval or generation anywhere else.

- [ ] **Step 6: Restrict evaluator injection**

Keep evaluator injection only in a private helper used by unit tests. Product CLI always calls `_run_real_ragas()`.

- [ ] **Step 7: Add failure behavior test**

```python
def test_failed_case_marks_run_failed_without_report(
    tmp_path,
    monkeypatch,
    valid_dataset,
):
    def fail_execution(*args, **kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(run, "run_rag_with_trace", fail_execution)

    with pytest.raises(RuntimeError, match="generation failed"):
        run_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            output_root=tmp_path,
            run_id="failed-run",
        )

    run_dir = tmp_path / "failed-run"
    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "failed"
    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "REPORT.md").exists()
```

- [ ] **Step 8: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_real_eval_runner.py tests/test_retrieval_evals.py -q
git add evals/run.py evals/ragas_adapter.py tests/test_real_eval_runner.py tests/test_retrieval_evals.py
git commit -m "01_RAG run only real production evaluations"
```

---

### Task 9: Statistical Comparison and Production Report

**Files:**

- Create: `01_RAG/evals/statistics.py`
- Create: `01_RAG/tests/test_eval_statistics.py`
- Modify: `01_RAG/evals/report.py`
- Modify: `01_RAG/tests/test_retrieval_evals.py`

**Interfaces:**

- Produces:

```python
@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float

def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 42,
) -> ConfidenceInterval: ...

def publish_completed_run(
    run_dir: Path,
    *,
    baseline_run: Path | None = None,
) -> None: ...
```

- [ ] **Step 1: Write deterministic bootstrap tests**

```python
def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap_delta([0.5, 0.6], [0.6, 0.7], seed=42)
    second = paired_bootstrap_delta([0.5, 0.6], [0.6, 0.7], seed=42)
    assert first == second
    assert first.estimate == pytest.approx(0.1)


def test_paired_bootstrap_rejects_unpaired_cases():
    with pytest.raises(ValueError, match="相同 Case"):
        paired_bootstrap_delta([0.5], [0.5, 0.6])
```

- [ ] **Step 2: Write report-content test**

Create completed fixture data and assert the report contains:

```text
BM25 Recall@50
Dense Recall@50
RRF Recall@80
Cross-Encoder NDCG@15
Final Context Recall@6
Final Context NDCG@6
Faithfulness
Answer Correctness
95% CI
退化样本
```

- [ ] **Step 3: Run tests to verify failure**

```bash
cd 01_RAG
pytest tests/test_eval_statistics.py tests/test_retrieval_evals.py -q
```

- [ ] **Step 4: Implement summary dimensions**

Aggregate by:

- Global.
- Split.
- Category.
- Query type.
- Expected behavior.

Behavior metrics:

- `abstain_accuracy`: fraction of `abstain` Cases that return the configured evidence-insufficient response.
- `deny_accuracy`: fraction of `deny` Cases that return access denial.
- `permission_leakage`: number of restricted qrels matched by any returned Candidate or exposed in the answer; release requirement is zero.

For each metric include average, count, and 95% bootstrap CI.

- [ ] **Step 5: Implement baseline gates**

Default gates:

```text
all Test cases completed
permission leakage == 0
final_context Recall@6 delta >= -0.02
final_context NDCG@6 delta >= -0.02
Faithfulness delta >= -0.02
Answer Correctness delta >= -0.02
abstain/deny accuracy does not regress
P95 latency ratio <= 1.20
```

Intermediate stages are diagnostic and do not independently block release.

- [ ] **Step 6: Prevent publishing incomplete runs**

`publish_completed_run()` checks:

- Manifest status is `completed`.
- Expected Case count equals completed Case count.
- No failed Case.
- All seven stage metrics exist for every answer Case.
- Test has at least 200 cases in release mode.

- [ ] **Step 7: Run tests and commit**

```bash
cd 01_RAG
pytest tests/test_eval_statistics.py tests/test_retrieval_evals.py -q
git add evals/statistics.py evals/report.py tests/test_eval_statistics.py tests/test_retrieval_evals.py
git commit -m "01_RAG report staged evaluation release gates"
```

---

### Task 10: Remove Legacy Metrics and Update Teaching Docs

**Files:**

- Modify: `01_RAG/evals/retrieval_metrics.py`
- Modify: `01_RAG/evals/run.py`
- Delete: `01_RAG/evals/dataset.jsonl`
- Modify: `01_RAG/05_rag_ragas_evaluation_design.md`
- Modify: `01_RAG/LEARNING_GUIDE.md`
- Modify: `01_RAG/README.md`
- Modify: `01_RAG/tests/test_retrieval_evals.py`

**Interfaces:**

- Removes legacy single-list evaluation and dry-run product path.

- [ ] **Step 1: Search for legacy references**

```bash
cd 01_RAG
rg -n "dry-run|_call_dry_run|dry_run_fixture|dataset.jsonl|retrieval_recall_at_5|DEFAULT_RETRIEVAL_K" .
```

- [ ] **Step 2: Remove legacy code and dataset**

Remove:

- `dry_run` arguments and helper functions.
- Legacy default dataset path.
- Legacy `build_retrieval_metric_fields()` once no callers remain.
- Documentation that presents source-level Recall@5 as the production metric.

Do not remove historical Git commits or the approved spec.

- [ ] **Step 3: Update commands**

Document only:

```bash
python -m evals.run --split dev --run-id candidate-v1
python -m evals.run --split test --run-id release-v1 --baseline-run evals/results/baseline-v1
```

Explain that empty/unreviewed datasets or missing stages make the command fail.

- [ ] **Step 4: Update teaching explanation**

Teach the two reporting layers:

- All four IR metrics are retained per stage for diagnosis.
- The release summary highlights the eight approved metrics.
- End-to-end RAGAS does not replace staged retrieval evaluation.
- There is no weighted “RAG total score.”

- [ ] **Step 5: Run documentation and reference checks**

```bash
cd 01_RAG
rg -n "dry-run|_call_dry_run|dry_run_fixture|evals/dataset.jsonl" evals rag tests README.md LEARNING_GUIDE.md 05_rag_ragas_evaluation_design.md
```

Expected: no active product or teaching references remain. Test names may mention that the CLI rejects `--dry-run`.

- [ ] **Step 6: Run focused tests and commit**

```bash
cd 01_RAG
pytest tests/test_retrieval_trace.py tests/test_qrels.py tests/test_stage_metrics.py tests/test_eval_checkpoint.py tests/test_real_eval_runner.py tests/test_eval_statistics.py tests/test_retrieval_evals.py -q
git add evals rag tests README.md LEARNING_GUIDE.md 05_rag_ragas_evaluation_design.md
git commit -m "01_RAG replace legacy RAG evaluation workflow"
```

Before committing, inspect `git diff --cached --name-status` and unstage unrelated files.

---

### Task 11: Real Integration Verification

**Files:**

- Create: `01_RAG/tests/test_real_evaluation_integration.py`
- Modify only if marker registration is needed: `01_RAG/pytest.ini`

**Interfaces:**

- Validates the assembled real-only system.

- [ ] **Step 1: Add explicitly gated real integration test**

The test runs only when:

```text
RUN_REAL_RAG_EVAL=1
```

It loads the first reviewed Case from `evals/datasets/dev.jsonl` and fails with a clear message when Dev is empty.

It must use:

- Real ES index.
- Real Embedding.
- Real Reranker.
- Real generation LLM.
- Real RAGAS Judge.

It asserts:

```python
assert trace.missing_required_stages() == []
assert answer
assert ragas_result["faithfulness"] is not None
assert stage_result["final_context_parent"]["ndcg_at_k"] is not None
assert manifest["status"] == "completed"
```

- [ ] **Step 2: Ensure integration output is isolated**

Use a temporary output root outside `evals/results`. The integration test must not create a production release report or mutate Test data.

- [ ] **Step 3: Run offline tests**

```bash
cd 01_RAG
pytest tests/ -q -k "not slow and not real_eval"
```

Expected: PASS.

- [ ] **Step 4: Run real integration test**

```bash
cd 01_RAG
RUN_REAL_RAG_EVAL=1 pytest tests/test_real_evaluation_integration.py -q
```

Expected: PASS with one real Case and seven real stages.

- [ ] **Step 5: Run a real Dev evaluation**

After reviewed Dev cases exist:

```bash
cd 01_RAG
python -m evals.run --split dev --run-id first-real-dev
```

Expected:

- Run status `completed`.
- No failed Cases.
- Seven stage metrics per answer Case.
- Real RAGAS values.
- `summary.json` and `REPORT.md` exist.

- [ ] **Step 6: Verify no fake path remains**

```bash
cd 01_RAG
rg -n "dry_run|dry-run|fixture.*1\\.0|ragas_mode" evals rag
```

Expected: no matches.

- [ ] **Step 7: Run final repository checks**

```bash
git status --short
git diff --check
git log --oneline -12
```

Confirm no secrets, generated reports, model files, ES data, or unrelated workspace changes are staged.

- [ ] **Step 8: Commit integration coverage**

```bash
git add 01_RAG/tests/test_real_evaluation_integration.py 01_RAG/pytest.ini
git commit -m "01_RAG verify real staged RAG evaluation"
```

If `pytest.ini` was not created or modified, do not include it in `git add`.

---

## Execution Gates

### Gate A: Evaluation Foundation

Tasks 1–5 can execute against the current repository:

- Trace contract.
- Dataset/qrel schema.
- Stable evidence resolver.
- Staged IR metrics.
- Checkpoint/fingerprint.

Gate A produces no official quality report.

### Gate B: Seven Real Retrieval Stages

Task 6 requires the Elasticsearch hybrid path plus real:

- Child Cross-Encoder.
- Business feature fusion.
- Parent aggregation/diversity.
- Final Token Budget context.

Missing stages block Tasks 7–11 from publishing formal results.

### Gate C: Human Gold Dataset

Software tests can pass with empty split files, but:

- Real Dev evaluation requires reviewed Dev cases.
- Real Test release gating requires at least 200 reviewed Test cases.
- No agent may invent qrels to bypass this gate.

## Final Verification

The feature is complete only when:

1. No product dry-run path exists.
2. Every answer Case has seven real stage snapshots.
3. Every stage computes Recall/NDCG/MRR/Hit from reviewed qrels.
4. Generation and RAGAS use the traced final context.
5. Failed runs cannot publish official reports.
6. Train/Dev/Test are physically and semantically isolated.
7. Candidate/Baseline reports include paired 95% confidence intervals.
8. A real Dev run completes successfully.
9. Test release gating remains blocked until at least 200 reviewed Test cases exist.
