"""Citation-grounded report writing skill and local citation audit helpers."""
from __future__ import annotations

import re

from agents.schemas import Citation
from runtime_skills.loader import load_skill_prompt

CITATION_REPORT_WRITING_SKILL = load_skill_prompt("citation-report-writing")

_CITATION = re.compile(r"\[\^(\d+)\]")
_REFERENCE_DEF = re.compile(r"^\[\^(\d+)\]:\s*(\S+)", re.MULTILINE)


def extract_citation_ids(markdown: str) -> set[int]:
    """Extract footnote ids used anywhere in markdown."""
    return {int(match) for match in _CITATION.findall(markdown)}


def extract_reference_ids(markdown: str) -> set[int]:
    """Extract footnote ids defined in the reference section."""
    return {int(match) for match, _url in _REFERENCE_DEF.findall(markdown)}


def validate_citation_ids(markdown: str, evidence_count: int) -> list[str]:
    """Return citation id issues that can be checked without an LLM."""
    issues: list[str] = []
    for idx in sorted(extract_citation_ids(markdown)):
        if idx < 1 or idx > evidence_count:
            issues.append(f"unknown citation [^{idx}]; evidence_count={evidence_count}")
    return issues


def build_missing_reference_entries(markdown: str, citations: list[Citation]) -> list[str]:
    """Build reference entries for cited evidence ids missing from markdown."""
    used = extract_citation_ids(markdown)
    defined = extract_reference_ids(markdown)
    by_idx = {citation.idx: citation for citation in citations}

    missing = []
    for idx in sorted(used - defined):
        citation = by_idx.get(idx)
        if citation is not None:
            missing.append(f"[^{idx}]: {citation.source_url}")
    return missing
