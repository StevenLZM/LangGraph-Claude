from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

from evals.models import EvidenceQrel
from rag.retrieval_trace import CandidateSnapshot


@dataclass(frozen=True)
class EvidenceMatch:
    evidence_id: str
    grade: int


def normalize_evidence_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def hash_evidence_text(text: str) -> str:
    normalized = normalize_evidence_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def page_ranges_overlap(left: str, right: str) -> bool:
    left_pages = _page_numbers(left)
    right_pages = _page_numbers(right)
    return bool(left_pages and right_pages and left_pages.intersection(right_pages))


def match_candidate(
    candidate: CandidateSnapshot,
    qrels: Sequence[EvidenceQrel],
) -> tuple[EvidenceMatch, ...]:
    candidate_source = _source_basename(candidate.source)
    candidate_text = normalize_evidence_text(candidate.page_content)
    matches: list[EvidenceMatch] = []

    for qrel in qrels:
        expected_hash = hash_evidence_text(qrel.evidence_text)
        if qrel.evidence_hash != expected_hash:
            raise ValueError(
                f"qrel {qrel.evidence_id} 的 evidence_hash 与 evidence_text 不一致"
            )
        if candidate_source != _source_basename(qrel.source):
            continue
        if not page_ranges_overlap(candidate.page_range, qrel.page_range):
            continue
        evidence_text = normalize_evidence_text(qrel.evidence_text)
        if not evidence_text or evidence_text not in candidate_text:
            continue
        matches.append(
            EvidenceMatch(
                evidence_id=qrel.evidence_id,
                grade=qrel.grade,
            )
        )
    return tuple(matches)


def _source_basename(source: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(source)).strip()
    return PurePosixPath(normalized.replace("\\", "/")).name


def _page_numbers(value: str) -> set[int]:
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("–", "-").replace("—", "-").replace("至", "-")
    pages: set[int] = set()
    for match in re.finditer(r"(\d+)(?:\s*-\s*(\d+))?", normalized):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start:
            start, end = end, start
        pages.update(range(start, end + 1))
    return pages
