from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self


MANIFEST_FILE = "run_manifest.json"
CASE_RESULTS_FILE = "case_results.partial.jsonl"
FAILED_CASES_FILE = "failed_cases.jsonl"


@dataclass(frozen=True)
class RunFingerprint:
    dataset_sha256: str
    corpus_version: str
    index_version: str
    git_commit: str
    config_sha256: str


class EvaluationCheckpoint:
    def __init__(
        self,
        run_dir: Path,
        fingerprint: RunFingerprint,
        manifest: dict[str, Any],
        completed_case_ids: set[str],
    ) -> None:
        self.run_dir = Path(run_dir)
        self.fingerprint = fingerprint
        self._manifest = dict(manifest)
        self.completed_case_ids = set(completed_case_ids)

    @classmethod
    def create(
        cls,
        run_dir: Path,
        fingerprint: RunFingerprint,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> Self:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / MANIFEST_FILE
        if manifest_path.exists():
            raise FileExistsError(f"运行目录已包含检查点: {run_dir}")

        manifest = {
            "schema_version": 1,
            "status": "running",
            "fingerprint": asdict(fingerprint),
            "completed_case_count": 0,
            "failed_case_count": 0,
            **dict(run_metadata or {}),
        }
        _write_json_atomic(manifest_path, manifest)
        (run_dir / CASE_RESULTS_FILE).touch(exist_ok=False)
        (run_dir / FAILED_CASES_FILE).touch(exist_ok=False)
        return cls(run_dir, fingerprint, manifest, set())

    @classmethod
    def resume(
        cls,
        run_dir: Path,
        fingerprint: RunFingerprint,
    ) -> Self:
        run_dir = Path(run_dir)
        manifest_path = run_dir / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(f"缺少运行检查点: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != asdict(fingerprint):
            raise ValueError("运行指纹不一致，不能恢复该评测")

        completed_case_ids = {
            str(result["id"])
            for result in _read_jsonl(run_dir / CASE_RESULTS_FILE)
            if str(result.get("id") or "").strip()
        }
        expected_count = int(manifest.get("completed_case_count", 0))
        if expected_count != len(completed_case_ids):
            raise ValueError("检查点 Case 数量与结果文件不一致")
        return cls(run_dir, fingerprint, manifest, completed_case_ids)

    @property
    def can_publish(self) -> bool:
        return (
            self._manifest.get("status") == "completed"
            and int(self._manifest.get("failed_case_count", 0)) == 0
        )

    @property
    def status(self) -> str:
        return str(self._manifest.get("status", "unknown"))

    def append_case(self, result: dict[str, Any]) -> None:
        case_id = str(result.get("id") or "").strip()
        if not case_id:
            raise ValueError("Case 结果必须包含非空 id")
        if case_id in self.completed_case_ids:
            raise ValueError(f"重复 Case 结果: {case_id}")

        _append_jsonl(self.run_dir / CASE_RESULTS_FILE, result)
        self.completed_case_ids.add(case_id)
        self._manifest["completed_case_count"] = len(self.completed_case_ids)
        self._persist_manifest()

    def append_failure(self, failure: dict[str, Any]) -> None:
        case_id = str(failure.get("id") or "").strip()
        if not case_id:
            raise ValueError("失败记录必须包含非空 id")
        _append_jsonl(self.run_dir / FAILED_CASES_FILE, failure)
        self._manifest["failed_case_count"] = (
            int(self._manifest.get("failed_case_count", 0)) + 1
        )
        self._persist_manifest()

    def mark_failed(self) -> None:
        self._manifest["status"] = "failed"
        self._persist_manifest()

    def mark_completed(self) -> None:
        if int(self._manifest.get("failed_case_count", 0)):
            raise ValueError("存在失败 Case，不能把运行标记为 completed")
        self._manifest["status"] = "completed"
        self._persist_manifest()

    def _persist_manifest(self) -> None:
        _write_json_atomic(self.run_dir / MANIFEST_FILE, self._manifest)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    with path.open("ab") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary_path.open("wb") as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少检查点结果文件: {path}")
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} 第 {line_number} 行必须是 JSON 对象")
        values.append(value)
    return values
