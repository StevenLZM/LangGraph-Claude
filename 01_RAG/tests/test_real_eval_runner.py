import hashlib
import json

import pytest
from langchain_core.documents import Document

from evals.checkpoint import RunFingerprint
from evals.qrels import hash_evidence_text


def _case(case_id: str = "dev-1") -> dict:
    return {
        "id": case_id,
        "category": "precise",
        "query_type": "keyword",
        "question": "产品保修多久？",
        "reference": "产品保修期为 12 个月。",
        "expected_behavior": "answer",
        "auth_context": {},
        "qrels": [
            {
                "evidence_id": f"{case_id}-evidence",
                "source": "manual.pdf",
                "page_range": "3",
                "section": "保修政策",
                "evidence_text": "产品保修期为 12 个月",
                "evidence_hash": hash_evidence_text("产品保修期为 12 个月"),
                "grade": 3,
            }
        ],
    }


@pytest.fixture
def valid_dataset(tmp_path):
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    contents = {
        "train": "\n",
        "dev": json.dumps(_case(), ensure_ascii=False) + "\n",
        "test": "\n",
    }
    for split, content in contents.items():
        (dataset_dir / f"{split}.jsonl").write_text(content, encoding="utf-8")
    manifest = {
        "version": "test-v1",
        "status": "reviewed",
        "corpus_version": "corpus-v1",
        "split_counts": {"train": 0, "dev": 1, "test": 0},
        "split_sha256": {
            split: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for split, content in contents.items()
        },
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    return dataset_dir


def _patch_successful_probes(monkeypatch, run):
    monkeypatch.setattr(run, "probe_elasticsearch", lambda: "index-v1")
    monkeypatch.setattr(run, "probe_parent_store", lambda: True)
    monkeypatch.setattr(run, "probe_embedding", lambda: True)
    monkeypatch.setattr(run, "probe_reranker", lambda: True)
    monkeypatch.setattr(run, "probe_generation_llm", lambda: True)
    monkeypatch.setattr(run, "probe_ragas_judge", lambda: True)
    monkeypatch.setattr(
        run,
        "probe_trace_stage_names",
        lambda: set(run.REQUIRED_EVAL_STAGES),
    )


def test_cli_has_no_dry_run_option():
    from evals.run import build_parser

    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--dry-run" not in option_strings
    assert "--split" in option_strings


def test_preflight_rejects_missing_required_stage(monkeypatch, valid_dataset):
    from evals import run

    _patch_successful_probes(monkeypatch, run)
    monkeypatch.setattr(run, "probe_trace_stage_names", lambda: {"bm25_child"})

    with pytest.raises(RuntimeError, match="缺少阶段"):
        run.preflight_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            release_mode=False,
        )


def test_preflight_rejects_unavailable_judge(monkeypatch, valid_dataset):
    from evals import run

    _patch_successful_probes(monkeypatch, run)
    monkeypatch.setattr(run, "probe_ragas_judge", lambda: False)

    with pytest.raises(RuntimeError, match="Judge"):
        run.preflight_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            release_mode=False,
        )


def test_failed_case_marks_run_failed_without_report(
    tmp_path,
    monkeypatch,
    valid_dataset,
):
    from evals import run

    fingerprint = RunFingerprint(
        dataset_sha256="dataset",
        corpus_version="corpus-v1",
        index_version="index-v1",
        git_commit="deadbeef",
        config_sha256="config",
    )
    monkeypatch.setattr(run, "preflight_real_evaluation", lambda **_: fingerprint)

    def fail_execution(*args, **kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(run, "run_rag_with_trace", fail_execution)

    with pytest.raises(RuntimeError, match="generation failed"):
        run.run_real_evaluation(
            split="dev",
            dataset_dir=valid_dataset,
            output_root=tmp_path,
            run_id="failed-run",
        )

    run_dir = tmp_path / "failed-run"
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert not (run_dir / "summary.json").exists()
    assert not (run_dir / "REPORT.md").exists()


def test_answer_case_uses_one_production_execution_for_ir_and_ragas(
    tmp_path,
    monkeypatch,
    valid_dataset,
):
    from evals import run
    from rag.chain import RagExecution
    from rag.retrieval_trace import StageRecorder

    fingerprint = RunFingerprint(
        dataset_sha256="dataset",
        corpus_version="corpus-v1",
        index_version="index-v1",
        git_commit="deadbeef",
        config_sha256="config",
    )
    monkeypatch.setattr(run, "preflight_real_evaluation", lambda **_: fingerprint)
    child = Document(
        page_content="产品保修期为 12 个月。",
        metadata={
            "doc_id": "doc-1",
            "child_id": "child-1",
            "parent_id": "parent-1",
            "source": "manual.pdf",
            "page_range": "3",
            "section_path": "保修政策",
            "bm25_score": 1.0,
            "dense_score": 1.0,
            "rrf_score": 1.0,
            "rerank_score": 1.0,
            "business_score": 1.0,
        },
    )
    parent = Document(
        page_content=child.page_content,
        metadata={
            **child.metadata,
            "evidence_id": "S1",
            "mmr_score": 1.0,
            "context_score": 1.0,
        },
    )
    recorder = StageRecorder(trace_id="trace-1", original_query="产品保修多久？")
    stage_docs = {
        "bm25_child": child,
        "dense_child": child,
        "rrf_child": child,
        "cross_encoder_child": child,
        "business_fused_child": child,
        "diversified_parent": parent,
        "final_context_parent": parent,
    }
    for name, document in stage_docs.items():
        recorder.record(
            name=name,
            documents=[document],
            configured_k=1,
            score_type=(
                "mmr_score"
                if name == "diversified_parent"
                else "context_score"
                if name == "final_context_parent"
                else {
                    "bm25_child": "bm25_score",
                    "dense_child": "dense_score",
                    "rrf_child": "rrf_score",
                    "cross_encoder_child": "rerank_score",
                    "business_fused_child": "business_score",
                }[name]
            ),
            latency_ms=1.0,
        )

    calls = {"rag": 0, "judge": 0}

    def fake_run(*args, **kwargs):
        calls["rag"] += 1
        return RagExecution(
            answer="产品保修期为 12 个月。[S1]",
            source_documents=(parent,),
            trace=recorder.trace,
            generation_latency_ms=2.0,
        )

    def fake_judge(rows, metric_names):
        calls["judge"] += 1
        assert rows[0]["retrieved_contexts"] == [parent.page_content]
        return [{**rows[0], **{name: 1.0 for name in metric_names}}]

    monkeypatch.setattr(run, "run_rag_with_trace", fake_run)

    run_dir = run._run_real_evaluation_impl(
        split="dev",
        dataset_dir=valid_dataset,
        output_root=tmp_path,
        run_id="one-pass",
        baseline_run=None,
        resume=False,
        ragas_evaluator=fake_judge,
    )

    assert calls == {"rag": 1, "judge": 1}
    result = json.loads(
        (run_dir / "case_results.jsonl").read_text(encoding="utf-8")
    )
    assert set(result["stage_metrics"]) == set(run.REQUIRED_EVAL_STAGES)
    assert result["ragas_metrics"]["faithfulness"] == 1.0
