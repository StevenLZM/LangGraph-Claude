"""Runtime LLM skill contracts.

These tests keep the skills testable without calling a real LLM. The prompts
define behavior for the model; the helpers enforce the parts we can verify
locally.
"""
from __future__ import annotations

from pathlib import Path

from agents.schemas import Citation, Evidence, EvidenceQuality
from runtime_skills.citation_report_writing import (
    CITATION_REPORT_WRITING_SKILL,
    build_missing_reference_entries,
    validate_citation_ids,
)
from runtime_skills.evidence_grounded_research import (
    EVIDENCE_GROUNDED_RESEARCH_SKILL,
    estimate_subquestion_coverage,
    format_coverage_guardrails,
)
from runtime_skills.loader import load_skill_prompt


def _evidence(
    *,
    url: str,
    support_level: str,
    confidence: float,
    sq: str = "sq1",
) -> Evidence:
    return Evidence(
        sub_question_id=sq,
        source_type="web",
        source_url=url,
        snippet="source text",
        relevance_score=confidence,
        quality=EvidenceQuality(
            support_level=support_level,
            source_authority="primary",
            extracted_claim="claim from source",
            confidence=confidence,
        ),
    )


def test_evidence_quality_limits_background_only_coverage():
    evidence = [
        _evidence(url="https://a.example", support_level="background", confidence=0.9),
        _evidence(url="https://b.example", support_level="irrelevant", confidence=0.9),
    ]

    assert estimate_subquestion_coverage("sq1", evidence) == 50


def test_evidence_quality_scores_direct_support_higher_than_indirect():
    evidence = [
        _evidence(url="https://a.example", support_level="direct", confidence=0.8),
        _evidence(url="https://b.example", support_level="indirect", confidence=0.8),
    ]

    assert estimate_subquestion_coverage("sq1", evidence) == 90


def test_validate_citation_ids_rejects_unknown_footnotes():
    issues = validate_citation_ids("结论一[^1]。错误引用[^99]。", evidence_count=3)

    assert issues == ["unknown citation [^99]; evidence_count=3"]


def test_build_missing_reference_entries_uses_backend_citations():
    markdown = "# 报告\n\n结论一[^1]，结论二[^2]。\n\n## 引用\n[^1]: https://a.example"
    citations = [
        Citation(idx=1, source_url="https://a.example"),
        Citation(idx=2, source_url="https://b.example"),
    ]

    assert build_missing_reference_entries(markdown, citations) == ["[^2]: https://b.example"]


def test_reflector_and_writer_prompts_include_runtime_skills():
    from prompts.templates import REFLECTOR_SYSTEM, WRITER_SYSTEM

    assert "Skill: evidence-grounded-research" in REFLECTOR_SYSTEM
    assert "runtime 已提供的 quality metadata" in REFLECTOR_SYSTEM
    assert "不要从零重新计算原始证据质量" in REFLECTOR_SYSTEM
    assert "Skill: citation-report-writing" in WRITER_SYSTEM


def test_runtime_skill_prompts_are_loaded_from_markdown_skill_files():
    project_root = Path(__file__).resolve().parents[1]
    assert (project_root / "skills/evidence-grounded-research/SKILL.md").is_file()
    assert (project_root / "skills/citation-report-writing/SKILL.md").is_file()

    assert EVIDENCE_GROUNDED_RESEARCH_SKILL == load_skill_prompt("evidence-grounded-research")
    assert CITATION_REPORT_WRITING_SKILL == load_skill_prompt("citation-report-writing")
    assert not EVIDENCE_GROUNDED_RESEARCH_SKILL.startswith("---")


def test_researcher_results_get_default_quality_metadata():
    from agents._researcher_base import _to_evidence

    evidence = _to_evidence(
        [
            {
                "source_url": "https://a.example",
                "snippet": "This source directly explains the answer.",
                "relevance_score": 0.82,
            }
        ],
        source_type="web",
        sub_question_id="sq1",
    )

    assert evidence[0].quality is not None
    assert evidence[0].quality.support_level == "direct"
    assert evidence[0].quality.extracted_claim == "This source directly explains the answer."


def test_reflector_evidence_summary_includes_quality_metadata():
    from agents.reflector import _format_evidence_summary

    text = _format_evidence_summary(
        [
            _evidence(url="https://a.example", support_level="direct", confidence=0.8),
        ]
    )

    assert "support=direct" in text
    assert "confidence=0.80" in text


def test_evidence_skill_outputs_structured_coverage_guardrails():
    text = format_coverage_guardrails(
        ["sq1", "sq2"],
        [
            _evidence(url="https://a.example", support_level="background", confidence=0.9, sq="sq1"),
            _evidence(url="https://b.example", support_level="direct", confidence=0.8, sq="sq2"),
        ],
    )

    assert "sq1: estimated_coverage=50" in text
    assert "sq2: estimated_coverage=80" in text
