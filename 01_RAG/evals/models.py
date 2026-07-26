from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Self


Split = Literal["train", "dev", "test"]
ExpectedBehavior = Literal["answer", "abstain", "deny"]
VALID_SPLITS: tuple[Split, ...] = ("train", "dev", "test")
VALID_BEHAVIORS: tuple[ExpectedBehavior, ...] = ("answer", "abstain", "deny")


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
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        grade = value.get("grade")
        if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
            raise ValueError("qrel grade 必须是 0 到 3 的整数")

        fields = {
            name: _required_text(value, name, owner="qrel")
            for name in (
                "evidence_id",
                "source",
                "page_range",
                "section",
                "evidence_text",
                "evidence_hash",
            )
        }
        return cls(**fields, grade=grade)


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    split: Split
    category: str
    query_type: str
    question: str
    reference: str
    expected_behavior: ExpectedBehavior
    auth_context: dict[str, Any]
    qrels: tuple[EvidenceQrel, ...]

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        split: str,
    ) -> Self:
        if split not in VALID_SPLITS:
            raise ValueError(f"未知 split: {split}")
        behavior = value.get("expected_behavior")
        if behavior not in VALID_BEHAVIORS:
            raise ValueError(
                "expected_behavior 必须是 answer、abstain 或 deny"
            )
        auth_context = value.get("auth_context")
        if not isinstance(auth_context, dict):
            raise ValueError("auth_context 必须是对象")

        raw_qrels = value.get("qrels")
        if not isinstance(raw_qrels, list):
            raise ValueError("qrels 必须是数组")
        qrels = tuple(EvidenceQrel.from_dict(item) for item in raw_qrels)
        evidence_ids = [qrel.evidence_id for qrel in qrels]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("同一 Case 内 evidence_id 必须唯一")
        if behavior == "answer" and not any(qrel.grade >= 2 for qrel in qrels):
            raise ValueError("answer Case 的 qrels 至少包含一个 grade >= 2 的证据")

        return cls(
            id=_required_text(value, "id", owner="case"),
            split=split,
            category=_required_text(value, "category", owner="case"),
            query_type=_required_text(value, "query_type", owner="case"),
            question=_required_text(value, "question", owner="case"),
            reference=_required_text(value, "reference", owner="case"),
            expected_behavior=behavior,
            auth_context=dict(auth_context),
            qrels=qrels,
        )


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    status: str
    corpus_version: str
    split_counts: dict[str, int]
    split_sha256: dict[str, str]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        counts = value.get("split_counts")
        hashes = value.get("split_sha256")
        if not isinstance(counts, dict) or not isinstance(hashes, dict):
            raise ValueError("manifest 必须包含 split_counts 和 split_sha256")
        return cls(
            version=_required_text(value, "version", owner="manifest"),
            status=_required_text(value, "status", owner="manifest"),
            corpus_version=_required_text(
                value,
                "corpus_version",
                owner="manifest",
            ),
            split_counts={
                split: int(counts.get(split, 0))
                for split in VALID_SPLITS
            },
            split_sha256={
                split: str(hashes.get(split, "")).strip()
                for split in VALID_SPLITS
            },
        )


def load_split(dataset_dir: Path, split: str) -> list[EvaluationCase]:
    if split not in VALID_SPLITS:
        raise ValueError(f"未知 split: {split}")
    path = Path(dataset_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"缺少数据集文件: {path}")

    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw_case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} 第 {line_number} 行不是合法 JSON"
            ) from exc
        if not isinstance(raw_case, dict):
            raise ValueError(f"{path.name} 第 {line_number} 行必须是 JSON 对象")
        cases.append(EvaluationCase.from_dict(raw_case, split=split))
    return cases


def validate_dataset(
    dataset_dir: Path,
    *,
    release_mode: bool,
) -> DatasetManifest:
    dataset_dir = Path(dataset_dir)
    split_cases = {
        split: load_split(dataset_dir, split)
        for split in VALID_SPLITS
    }
    _validate_unique_ids(split_cases)
    _validate_cross_split_questions(split_cases)

    counts = {
        split: len(split_cases[split])
        for split in VALID_SPLITS
    }
    hashes = {
        split: _sha256_file(dataset_dir / f"{split}.jsonl")
        for split in VALID_SPLITS
    }
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = DatasetManifest.from_dict(raw_manifest)
        if manifest.split_counts != counts:
            raise ValueError(
                f"manifest split_counts 与数据文件不一致: {counts}"
            )
        if manifest.split_sha256 != hashes:
            raise ValueError("manifest split_sha256 与数据文件不一致")
    else:
        manifest = DatasetManifest(
            version="unmanaged",
            status="unmanaged",
            corpus_version="unmanaged",
            split_counts=counts,
            split_sha256=hashes,
        )

    if release_mode and counts["test"] < 200:
        raise ValueError("正式 Test 评测至少 200 个已复核 Case")
    return manifest


def normalize_question_fingerprint(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(question)).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


def _validate_unique_ids(
    split_cases: Mapping[str, list[EvaluationCase]],
) -> None:
    owners: dict[str, str] = {}
    for split, cases in split_cases.items():
        for case in cases:
            previous = owners.get(case.id)
            if previous is not None:
                raise ValueError(
                    f"Case id 重复: {case.id} 同时出现在 {previous} 和 {split}"
                )
            owners[case.id] = split


def _validate_cross_split_questions(
    split_cases: Mapping[str, list[EvaluationCase]],
) -> None:
    owners: dict[str, tuple[str, str]] = {}
    for split, cases in split_cases.items():
        for case in cases:
            fingerprint = normalize_question_fingerprint(case.question)
            previous = owners.get(fingerprint)
            if previous is not None and previous[0] != split:
                raise ValueError(
                    "检测到跨 split 问题泄漏: "
                    f"{previous[1]} ({previous[0]}) 与 {case.id} ({split})"
                )
            owners[fingerprint] = (split, case.id)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(
    value: Mapping[str, Any],
    name: str,
    *,
    owner: str,
) -> str:
    text = str(value.get(name) or "").strip()
    if not text:
        raise ValueError(f"{owner} 的 {name} 不能为空")
    return text
