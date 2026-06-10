"""Evidence-grounded research skill.

The prompt is used by runtime LLM agents. The helpers keep the skill contract
testable for offline unit tests and deterministic fallbacks.
"""
from __future__ import annotations

from agents.schemas import Evidence, EvidenceQuality
from runtime_skills.loader import load_skill_prompt

EVIDENCE_GROUNDED_RESEARCH_SKILL = load_skill_prompt("evidence-grounded-research")

_SUPPORT_RANK = {
    "irrelevant": 0,
    "background": 1,
    "indirect": 2,
    "direct": 3,
}


def estimate_subquestion_coverage(sub_question_id: str, evidence: list[Evidence]) -> int:
    """Estimate coverage from quality metadata without calling an LLM.

    This is not a replacement for the reflector. It provides a deterministic
    guardrail: background-only material cannot look "sufficient" just because
    there are many results.
    """
    scoped = [ev for ev in evidence if ev.sub_question_id == sub_question_id]
    if not scoped:
        return 0

    direct = _count_support(scoped, "direct")
    indirect = _count_support(scoped, "indirect")
    background = _count_support(scoped, "background")

    if direct:
        return min(100, 80 + min(10, (direct - 1) * 5) + min(10, indirect * 10))
    if indirect:
        return min(75, 55 + min(20, (indirect - 1) * 10))
    if background:
        return 50
    return 0


def strongest_support_level(evidence: list[Evidence]) -> str:
    """Return the strongest support level in a list of evidence."""
    strongest = "irrelevant"
    for ev in evidence:
        level = getattr(ev.quality, "support_level", None) if ev.quality else None
        if _SUPPORT_RANK.get(level or "irrelevant", 0) > _SUPPORT_RANK[strongest]:
            strongest = level or strongest
    return strongest


def format_coverage_guardrails(sub_question_ids: list[str], evidence: list[Evidence]) -> str:
    """Format deterministic coverage estimates for reflector context."""
    if not sub_question_ids:
        return "(无 coverage guardrails)"
    lines = []
    for sub_question_id in sub_question_ids:
        lines.append(
            f"- {sub_question_id}: estimated_coverage="
            f"{estimate_subquestion_coverage(sub_question_id, evidence)}"
        )
    return "\n".join(lines)


def build_default_quality(snippet: str, relevance_score: float) -> EvidenceQuality:
    """Build deterministic quality metadata from tool scores.

    Tool scores are weaker than LLM judgment, but a conservative default keeps
    the downstream reflector from treating all results as unclassified.
    """
    score = max(0.0, min(1.0, float(relevance_score or 0.0)))
    if score >= 0.75:
        support_level = "direct"
    elif score >= 0.45:
        support_level = "indirect"
    elif score > 0:
        support_level = "background"
    else:
        support_level = "irrelevant"

    return EvidenceQuality(
        support_level=support_level,
        source_authority="unknown",
        extracted_claim=(snippet or "").strip()[:240],
        confidence=score,
    )


def _count_support(evidence: list[Evidence], support_level: str) -> int:
    return sum(1 for ev in evidence if ev.quality and ev.quality.support_level == support_level)
