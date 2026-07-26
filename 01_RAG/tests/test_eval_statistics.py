import json

import pytest

from rag.retrieval_trace import REQUIRED_EVAL_STAGES


def _stage_metrics() -> dict:
    metrics = {}
    configured_k = {
        "bm25_child": 50,
        "dense_child": 50,
        "rrf_child": 80,
        "cross_encoder_child": 15,
        "business_fused_child": 15,
        "diversified_parent": 8,
        "final_context_parent": 6,
    }
    for stage_name in REQUIRED_EVAL_STAGES:
        metrics[stage_name] = {
            "stage_name": stage_name,
            "configured_k": configured_k[stage_name],
            "actual_count": 1,
            "relevant_evidence_count": 1,
            "covered_evidence_count": 1,
            "recall_at_k": 1.0,
            "ndcg_at_k": 1.0,
            "mrr_at_k": 1.0,
            "hit_at_k": 1.0,
        }
    return metrics


def _write_completed_run(run_dir):
    run_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "split": "dev",
        "release_mode": False,
        "expected_case_count": 1,
        "completed_case_count": 1,
        "failed_case_count": 0,
        "fingerprint": {
            "dataset_sha256": "dataset",
            "corpus_version": "corpus-v1",
            "index_version": "index-v1",
            "git_commit": "deadbeef",
            "config_sha256": "config",
        },
    }
    result = {
        "id": "dev-1",
        "split": "dev",
        "category": "precise",
        "query_type": "keyword",
        "expected_behavior": "answer",
        "answer": "产品保修期为 12 个月。[S1]",
        "degraded_stages": [],
        "stage_metrics": _stage_metrics(),
        "ragas_metrics": {
            "context_precision": 1.0,
            "context_recall": 1.0,
            "faithfulness": 0.95,
            "answer_correctness": 0.93,
            "answer_relevancy": 0.90,
            "semantic_similarity": 0.92,
        },
        "behavior_metrics": {
            "abstain_accuracy": None,
            "deny_accuracy": None,
            "permission_leakage": 0,
        },
        "latency_ms": 120.0,
        "generation_latency_ms": 50.0,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (run_dir / "case_results.partial.jsonl").write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "failed_cases.jsonl").write_text("", encoding="utf-8")


def test_paired_bootstrap_is_reproducible():
    from evals.statistics import paired_bootstrap_delta

    first = paired_bootstrap_delta([0.5, 0.6], [0.6, 0.7], seed=42)
    second = paired_bootstrap_delta([0.5, 0.6], [0.6, 0.7], seed=42)

    assert first == second
    assert first.estimate == pytest.approx(0.1)


def test_paired_bootstrap_rejects_unpaired_cases():
    from evals.statistics import paired_bootstrap_delta

    with pytest.raises(ValueError, match="相同 Case"):
        paired_bootstrap_delta([0.5], [0.5, 0.6])


def test_completed_report_contains_production_stage_metrics(tmp_path):
    from evals.report import publish_completed_run

    run_dir = tmp_path / "candidate"
    _write_completed_run(run_dir)

    publish_completed_run(run_dir)

    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "BM25 Recall@50" in report
    assert "Dense Recall@50" in report
    assert "RRF Recall@80" in report
    assert "Cross-Encoder NDCG@15" in report
    assert "Final Context Recall@6" in report
    assert "Final Context NDCG@6" in report
    assert "Faithfulness" in report
    assert "Answer Correctness" in report
    assert "95% CI" in report
    assert "退化样本" in report
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert "total_score" not in summary
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "case_results.jsonl").exists()
    assert (run_dir / "stage_metrics.jsonl").exists()
    assert (run_dir / "ragas_results.jsonl").exists()


def test_incomplete_run_cannot_publish(tmp_path):
    from evals.report import publish_completed_run

    run_dir = tmp_path / "incomplete"
    _write_completed_run(run_dir)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "failed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="completed"):
        publish_completed_run(run_dir)

    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "REPORT.md").exists()
