"""Opt-in integration test for the complete real RAG evaluation stack."""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.slow,
    pytest.mark.real_eval,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_RAG_EVAL") != "1",
        reason="set RUN_REAL_RAG_EVAL=1 to run the real evaluation stack",
    ),
]


def test_first_reviewed_dev_case_completes_real_evaluation(tmp_path):
    from evals.models import load_split
    from evals.run import run_real_evaluation
    from rag.retrieval_trace import REQUIRED_EVAL_STAGES

    project_root = Path(__file__).resolve().parents[1]
    source_dataset = project_root / "evals" / "datasets"
    cases = load_split(source_dataset, "dev")
    assert cases, (
        "Dev split 为空：必须先由人工复核真实 reference 和 qrels，"
        "不能用脚本生成评测标签"
    )

    isolated_dataset = tmp_path / "datasets"
    isolated_dataset.mkdir()
    dev_content = json.dumps(
        asdict(cases[0]),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "\n"
    contents = {"train": "\n", "dev": dev_content, "test": "\n"}
    for split, content in contents.items():
        (isolated_dataset / f"{split}.jsonl").write_text(
            content,
            encoding="utf-8",
        )
    manifest = {
        "version": "real-integration-v1",
        "status": "reviewed",
        "corpus_version": json.loads(
            (source_dataset / "manifest.json").read_text(encoding="utf-8")
        )["corpus_version"],
        "split_counts": {"train": 0, "dev": 1, "test": 0},
        "split_sha256": {
            split: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for split, content in contents.items()
        },
    }
    (isolated_dataset / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    run_dir = run_real_evaluation(
        split="dev",
        dataset_dir=isolated_dataset,
        output_root=tmp_path / "results",
        run_id="real-integration",
    )

    result = json.loads(
        (run_dir / "case_results.jsonl").read_text(encoding="utf-8")
    )
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert set(result["stage_metrics"]) == set(REQUIRED_EVAL_STAGES)
    assert result["answer"]
    assert result["ragas_metrics"]["faithfulness"] is not None
    assert (
        result["stage_metrics"]["final_context_parent"]["ndcg_at_k"]
        is not None
    )
    assert run_manifest["status"] == "completed"
