import json

import pytest

from evals.checkpoint import EvaluationCheckpoint, RunFingerprint


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


def test_failed_run_cannot_publish_summary(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.mark_failed()

    assert checkpoint.can_publish is False
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "REPORT.md").exists()


def test_checkpoint_persists_completed_cases_for_resume(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.append_case({"id": "case-1", "score": 0.8})

    resumed = EvaluationCheckpoint.resume(tmp_path, _fingerprint("config-a"))

    assert resumed.completed_case_ids == {"case-1"}
    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["completed_case_count"] == 1


def test_checkpoint_rejects_duplicate_case_result(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.append_case({"id": "case-1"})

    with pytest.raises(ValueError, match="重复"):
        checkpoint.append_case({"id": "case-1"})


def test_run_with_failure_cannot_be_marked_completed(tmp_path):
    checkpoint = EvaluationCheckpoint.create(tmp_path, _fingerprint("config-a"))
    checkpoint.append_failure({"id": "case-1", "error": "generation failed"})

    with pytest.raises(ValueError, match="失败"):
        checkpoint.mark_completed()
