"""Manual review queue and JSONL persistence for eval results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REVIEW_ACTIONS = {"manual_review", "spot_check"}
VALID_LABELS = {
    "pass",
    "minor_issue",
    "major_issue",
    "unsafe_or_misleading",
    "unjudgeable",
}


def build_review_queue(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build pending manual-review tasks from eval records."""
    queue = []
    for rec in records:
        sampling = rec.get("sampling") or {}
        if sampling.get("review_action") not in REVIEW_ACTIONS:
            continue
        case = rec.get("case") or {}
        reasons = list(sampling.get("reasons") or [])
        queue.append({
            "case_id": case.get("id"),
            "thread_id": rec.get("thread_id"),
            "query": case.get("query"),
            "category": case.get("category"),
            "risk_level": sampling.get("risk_level"),
            "review_action": sampling.get("review_action"),
            "reasons": reasons,
            "overall": (rec.get("score") or {}).get("overall"),
            "report_path": rec.get("report_path"),
            "suggested_label": suggest_review_label(rec),
            "status": "pending",
        })
    return queue


def suggest_review_label(record: dict[str, Any]) -> str:
    """Suggest a starting label for human reviewers."""
    sampling = record.get("sampling") or {}
    reasons = set(sampling.get("reasons") or [])
    if "execution_error" in reasons:
        return "unjudgeable"
    if "low_overall_score" in reasons or "zero_evidence" in reasons:
        return "major_issue"
    if "citation_audit_issue" in reasons or "forced_completion" in reasons:
        return "major_issue"
    if "medium_overall_score" in reasons or "missing_subquestion_evidence" in reasons:
        return "minor_issue"
    return "pass"


def write_review_queue(queue: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


def append_manual_review(
    path: Path,
    *,
    case_id: str,
    reviewer: str,
    label: str,
    notes: str,
    root_cause: str,
) -> Path:
    if label not in VALID_LABELS:
        raise ValueError(f"invalid review label: {label}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case_id,
        "reviewer": reviewer,
        "label": label,
        "notes": notes,
        "root_cause": root_cause,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def load_manual_reviews(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
