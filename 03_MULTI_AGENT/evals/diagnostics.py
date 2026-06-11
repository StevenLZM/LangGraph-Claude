"""Diagnostics for the three-layer InsightLoop eval view.

The LLM judge scores final-answer quality. These helpers add deterministic
process and runtime signals so eval reports can guide manual spot checks.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _get(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _score_value(score: Any, key: str) -> int | None:
    if score is None:
        return None
    value = _get(score, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_process_quality(
    *,
    plan: list[Any] | None,
    evidence: list[Any] | None,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize planner/research/reflector/writer signals without an LLM."""
    plan = list(plan or [])
    evidence = list(evidence or [])
    state = state or {}

    plan_ids = [str(_get(sq, "id", "")) for sq in plan if _get(sq, "id", "")]
    evidence_per_subq = {sid: 0 for sid in plan_ids}
    source_counts: Counter[str] = Counter()

    for ev in evidence:
        sid = str(_get(ev, "sub_question_id", "") or "")
        if sid:
            evidence_per_subq[sid] = evidence_per_subq.get(sid, 0) + 1
        source = str(_get(ev, "source_type", "") or "unknown")
        source_counts[source] += 1

    citation_issues = list(state.get("citation_audit_issues") or [])
    missing_subquestions = [sid for sid in plan_ids if evidence_per_subq.get(sid, 0) == 0]
    next_action = state.get("next_action")

    return {
        "plan_size": len(plan),
        "evidence_count": len(evidence),
        "evidence_per_subq": evidence_per_subq,
        "missing_subquestions": missing_subquestions,
        "source_counts": dict(source_counts),
        "source_diversity": len(source_counts),
        "revision_count": int(state.get("revision_count") or 0),
        "next_action": next_action,
        "forced_completion": next_action == "force_complete",
        "missing_aspects": list(state.get("missing_aspects") or []),
        "coverage_by_subq": dict(state.get("coverage_by_subq") or {}),
        "citation_audit_issue_count": len(citation_issues),
        "citation_audit_issues": citation_issues,
    }


def build_retrieval_metrics(
    *,
    plan: list[Any] | None,
    evidence: list[Any] | None,
) -> dict[str, Any]:
    """Build RAG-style proxy metrics from subquestion/evidence structure."""
    plan = list(plan or [])
    evidence = list(evidence or [])

    plan_ids = [str(_get(sq, "id", "")) for sq in plan if _get(sq, "id", "")]
    planned_sources: set[str] = set()
    for sq in plan:
        sources = _get(sq, "recommended_sources", None) or ["web"]
        planned_sources.update(str(src) for src in sources if src)

    evidence_subquestions = {
        str(_get(ev, "sub_question_id", "") or "")
        for ev in evidence
        if _get(ev, "sub_question_id", "")
    }
    covered_sources = {
        str(_get(ev, "source_type", "") or "")
        for ev in evidence
        if _get(ev, "source_type", "")
    }

    planned_subq_count = len(plan_ids)
    covered_subq_count = sum(1 for sid in plan_ids if sid in evidence_subquestions)
    subquestion_recall = (
        round(covered_subq_count / planned_subq_count, 3)
        if planned_subq_count
        else 0.0
    )
    source_type_recall = (
        round(len(planned_sources & covered_sources) / len(planned_sources), 3)
        if planned_sources
        else 0.0
    )
    evidence_density = (
        round(len(evidence) / planned_subq_count, 2)
        if planned_subq_count
        else 0.0
    )

    return {
        "planned_subquestion_count": planned_subq_count,
        "covered_subquestion_count": covered_subq_count,
        "subquestion_recall": subquestion_recall,
        "planned_source_types": sorted(planned_sources),
        "covered_source_types": sorted(covered_sources),
        "source_type_recall": source_type_recall,
        "evidence_density": evidence_density,
    }


def build_runtime_health(
    *,
    error: str | None,
    elapsed_sec: float | None,
    slow_threshold_sec: float = 240.0,
) -> dict[str, Any]:
    """Summarize execution stability signals."""
    error_type = None
    if error:
        error_type = error.split(":", 1)[0].strip() or "Error"

    return {
        "status": "error" if error else "ok",
        "elapsed_sec": elapsed_sec,
        "slow_case": bool(elapsed_sec is not None and elapsed_sec >= slow_threshold_sec),
        "error_type": error_type,
    }


def _component(status: str, score: int, signals: list[str]) -> dict[str, Any]:
    return {"status": status, "score": max(0, min(100, score)), "signals": signals}


def build_component_quality(
    *,
    process_quality: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    runtime_health: dict[str, Any],
    score: dict[str, Any] | None,
) -> dict[str, Any]:
    """Score major Agent components with deterministic teaching heuristics."""
    plan_size = int(process_quality.get("plan_size") or 0)
    evidence_count = int(process_quality.get("evidence_count") or 0)
    citation_issues = int(process_quality.get("citation_audit_issue_count") or 0)

    planner_score = 100 if plan_size > 0 else 0
    planner_status = "pass" if planner_score == 100 else "fail"

    research_score = round(float(retrieval_metrics.get("subquestion_recall") or 0.0) * 100)
    if plan_size > 0 and evidence_count == 0:
        research_status = "fail"
    elif research_score < 80:
        research_status = "warn"
    else:
        research_status = "pass"

    if process_quality.get("forced_completion"):
        reflector_score = 60
        reflector_status = "warn"
    elif process_quality.get("missing_aspects"):
        reflector_score = 80
        reflector_status = "warn"
    else:
        reflector_score = 100
        reflector_status = "pass"

    citation_score = _score_value(score, "citation")
    writer_base = citation_score if citation_score is not None else 100
    writer_score = writer_base - citation_issues * 25
    writer_status = "fail" if writer_score < 60 else "warn" if writer_score < 80 else "pass"

    if runtime_health.get("status") == "error":
        runtime_score = 0
        runtime_status = "fail"
    elif runtime_health.get("slow_case"):
        runtime_score = 80
        runtime_status = "warn"
    else:
        runtime_score = 100
        runtime_status = "pass"

    return {
        "planner": _component(planner_status, planner_score, [f"plan_size={plan_size}"]),
        "research": _component(
            research_status,
            research_score,
            [
                f"subquestion_recall={retrieval_metrics.get('subquestion_recall', 0.0)}",
                f"evidence_count={evidence_count}",
            ],
        ),
        "reflector": _component(
            reflector_status,
            reflector_score,
            [
                f"revision_count={process_quality.get('revision_count', 0)}",
                f"forced_completion={process_quality.get('forced_completion', False)}",
            ],
        ),
        "writer": _component(
            writer_status,
            writer_score,
            [
                f"citation_score={citation_score}",
                f"citation_audit_issues={citation_issues}",
            ],
        ),
        "runtime": _component(
            runtime_status,
            runtime_score,
            [
                f"status={runtime_health.get('status')}",
                f"slow_case={runtime_health.get('slow_case')}",
            ],
        ),
    }


def build_sampling_decision(
    *,
    score: dict[str, Any] | None,
    process_quality: dict[str, Any],
    runtime_health: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic review reasons for no-feedback spot checks."""
    reasons: list[str] = []
    overall = _score_value(score, "overall")

    if runtime_health.get("status") == "error":
        reasons.append("execution_error")
    if overall is None and runtime_health.get("status") != "error":
        reasons.append("missing_judge_score")
    elif overall is not None and overall < 70:
        reasons.append("low_overall_score")
    elif overall is not None and overall < 80:
        reasons.append("medium_overall_score")

    if process_quality.get("citation_audit_issue_count", 0) > 0:
        reasons.append("citation_audit_issue")
    if process_quality.get("missing_subquestions"):
        reasons.append("missing_subquestion_evidence")
    if process_quality.get("forced_completion"):
        reasons.append("forced_completion")
    if runtime_health.get("slow_case"):
        reasons.append("slow_case")
    if process_quality.get("plan_size", 0) > 0 and process_quality.get("evidence_count", 0) == 0:
        reasons.append("zero_evidence")

    high_risk = {
        "execution_error",
        "missing_judge_score",
        "low_overall_score",
        "citation_audit_issue",
        "forced_completion",
        "zero_evidence",
    }
    medium_risk = {"medium_overall_score", "missing_subquestion_evidence", "slow_case"}

    if any(reason in high_risk for reason in reasons):
        risk_level = "high"
        review_action = "manual_review"
    elif any(reason in medium_risk for reason in reasons):
        risk_level = "medium"
        review_action = "spot_check"
    else:
        risk_level = "low"
        review_action = "random_sample"

    return {
        "risk_level": risk_level,
        "review_action": review_action,
        "reasons": reasons,
    }
