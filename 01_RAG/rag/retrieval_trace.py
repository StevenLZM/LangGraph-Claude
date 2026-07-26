from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from langchain_core.documents import Document


REQUIRED_EVAL_STAGES: tuple[str, ...] = (
    "bm25_child",
    "dense_child",
    "rrf_child",
    "cross_encoder_child",
    "business_fused_child",
    "diversified_parent",
    "final_context_parent",
)

CandidateLevel = Literal["child", "parent"]


@dataclass(frozen=True)
class CandidateSnapshot:
    rank: int
    level: CandidateLevel
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
    candidate_level: CandidateLevel
    configured_k: int
    latency_ms: float
    candidates: tuple[CandidateSnapshot, ...]


@dataclass
class EvaluationTrace:
    trace_id: str
    original_query: str
    rewritten_query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, StageSnapshot] = field(default_factory=dict)

    def missing_required_stages(self) -> list[str]:
        return [name for name in REQUIRED_EVAL_STAGES if name not in self.stages]

    def require_complete(self) -> None:
        missing = self.missing_required_stages()
        if missing:
            raise ValueError(f"评测 Trace 缺少阶段: {', '.join(missing)}")


class StageRecorder:
    def __init__(
        self,
        *,
        trace_id: str,
        original_query: str,
        rewritten_query: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._trace = EvaluationTrace(
            trace_id=str(trace_id),
            original_query=str(original_query),
            rewritten_query=str(rewritten_query),
            metadata=dict(metadata or {}),
        )

    @property
    def trace(self) -> EvaluationTrace:
        return self._trace

    def record(
        self,
        *,
        name: str,
        documents: Sequence[Document],
        configured_k: int,
        score_type: str,
        latency_ms: float,
    ) -> StageSnapshot:
        if name in self._trace.stages:
            raise ValueError(f"重复阶段: {name}")
        if configured_k < 0:
            raise ValueError("configured_k 不能小于 0")
        if latency_ms < 0:
            raise ValueError("latency_ms 不能小于 0")

        level = _candidate_level(name)
        candidates = tuple(
            _snapshot_document(
                document,
                rank=rank,
                level=level,
                score_type=score_type,
            )
            for rank, document in enumerate(documents, start=1)
        )
        stage = StageSnapshot(
            name=name,
            candidate_level=level,
            configured_k=int(configured_k),
            latency_ms=float(latency_ms),
            candidates=candidates,
        )
        self._trace.stages[name] = stage
        return stage


def _candidate_level(stage_name: str) -> CandidateLevel:
    if stage_name.endswith("_child"):
        return "child"
    if stage_name.endswith("_parent"):
        return "parent"
    raise ValueError(f"阶段名称必须以 _child 或 _parent 结尾: {stage_name}")


def _snapshot_document(
    document: Document,
    *,
    rank: int,
    level: CandidateLevel,
    score_type: str,
) -> CandidateSnapshot:
    metadata = dict(document.metadata or {})
    parent_id = _text(metadata.get("parent_id"))
    raw_child_id = _text(metadata.get("child_id"))
    child_id = raw_child_id or None

    if level == "child" and child_id is None:
        raise ValueError(f"Child 阶段第 {rank} 个候选缺少 child_id")
    if level == "parent" and not parent_id:
        raise ValueError(f"Parent 阶段第 {rank} 个候选缺少 parent_id")

    doc_id = _text(metadata.get("doc_id")) or parent_id or (child_id or "")
    raw_score = metadata.get(score_type)
    score = float(raw_score) if raw_score is not None else None
    return CandidateSnapshot(
        rank=rank,
        level=level,
        doc_id=doc_id,
        parent_id=parent_id,
        child_id=child_id,
        source=_text(metadata.get("source")),
        page_range=_text(metadata.get("page_range")),
        section=_text(metadata.get("section_path") or metadata.get("section")),
        score=score,
        score_type=str(score_type),
        page_content=str(document.page_content or ""),
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
