from pathlib import Path
import json
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_dataset_loader_requires_ragas_reference(tmp_path):
    from evals.run import load_dataset

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"reference":"产品保修期为 12 个月。"}\n',
        encoding="utf-8",
    )

    cases = load_dataset(dataset_path)

    assert cases[0]["reference"] == "产品保修期为 12 个月。"


def test_dataset_loader_rejects_missing_ragas_reference(tmp_path):
    import pytest

    from evals.run import load_dataset

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"expected_sources":["spec.pdf"]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference"):
        load_dataset(dataset_path)


def test_build_ragas_rows_preserves_retrieval_semantic_and_e2e_fields():
    from evals.ragas_adapter import build_ragas_row

    case = {
        "id": "case_001",
        "category": "precise",
        "question": "产品保修期多久？",
        "reference": "产品保修期为 12 个月。",
        "query_type": "keyword",
    }
    docs = [
        Document(
            page_content="产品保修期为 12 个月，从购买之日起计算。",
            metadata={"source": "spec.pdf", "parent_id": "parent-warranty"},
        )
    ]
    response = {"answer": "产品保修期为 12 个月。", "sources": docs}

    row = build_ragas_row(case, docs, response)

    assert row["id"] == "case_001"
    assert row["category"] == "precise"
    assert row["user_input"] == "产品保修期多久？"
    assert row["retrieved_contexts"] == ["产品保修期为 12 个月，从购买之日起计算。"]
    assert row["response"] == "产品保修期为 12 个月。"
    assert row["reference"] == "产品保修期为 12 个月。"
    assert row["retrieved_sources"] == ["spec.pdf"]


def test_ragas_result_aliases_are_normalized_to_project_metric_names():
    from evals.ragas_adapter import _score_rows_to_dicts

    rows = _score_rows_to_dicts(
        [
            {
                "llm_context_precision_with_reference": 0.8,
                "context_recall": 0.7,
                "user_input": "ignored",
            }
        ]
    )

    assert rows == [{"context_precision": 0.8, "context_recall": 0.7}]


def test_run_ragas_evaluation_writes_results_summary_and_report(tmp_path):
    from evals.run import run_evaluation

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"reference":"产品保修期为 12 个月。"}\n',
        encoding="utf-8",
    )

    def fake_retriever(case):
        assert case["question"] == "保修期多久？"
        return [
            Document(
                page_content="产品保修期为 12 个月。",
                metadata={"source": "spec.pdf", "parent_id": "parent-warranty"},
            )
        ]

    def fake_generator(case, docs):
        assert docs[0].page_content == "产品保修期为 12 个月。"
        return {"answer": "产品保修期为 12 个月。", "sources": docs}

    def fake_ragas_evaluator(rows, metric_names):
        assert metric_names == [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_correctness",
            "answer_relevancy",
            "semantic_similarity",
        ]
        assert rows[0]["user_input"] == "保修期多久？"
        return [
            {
                **rows[0],
                "context_precision": 1.0,
                "context_recall": 1.0,
                "faithfulness": 0.95,
                "answer_correctness": 0.93,
                "answer_relevancy": 0.9,
                "semantic_similarity": 0.92,
            }
        ]

    run_dir = run_evaluation(
        dataset_path=dataset_path,
        output_root=tmp_path / "results",
        retrieval_callable=fake_retriever,
        generation_callable=fake_generator,
        ragas_evaluator=fake_ragas_evaluator,
        run_id="unit-run",
    )

    results_path = run_dir / "ragas_results.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "REPORT.md"
    assert results_path.exists()
    assert summary_path.exists()
    assert report_path.exists()

    result = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    assert result["context_precision"] == 1.0
    assert result["semantic_similarity"] == 0.92

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["metrics"]["context_precision"]["average"] == 1.0
    assert summary["metrics"]["faithfulness"]["average"] == 0.95
    assert summary["metrics"]["answer_correctness"]["average"] == 0.93

    report = report_path.read_text(encoding="utf-8")
    assert "# 01_RAG RAGAS 评估报告" in report
    assert "Context Precision" in report
    assert "Answer Correctness" in report
    assert "Semantic Similarity" in report
    assert "Recall@5" not in report
    assert "MRR" not in report
    assert "Parent Hit" not in report
    assert "关键词" not in report


def test_packaged_eval_dataset_is_valid():
    from evals.run import load_dataset

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset.jsonl"

    cases = load_dataset(dataset_path)

    assert len(cases) >= 5
    assert {case["category"] for case in cases} >= {"conceptual", "precise", "time"}
    assert all(case.get("reference") for case in cases)


def test_run_evaluation_dry_run_uses_packaged_dataset(tmp_path):
    from evals.run import run_evaluation

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset.jsonl"

    run_dir = run_evaluation(
        dataset_path=dataset_path,
        output_root=tmp_path / "results",
        run_id="dry-run",
        dry_run=True,
    )

    summary = (run_dir / "summary.json").read_text(encoding="utf-8")
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert '"total":' in summary
    assert "01_RAG RAGAS 评估报告" in report
    assert "RAGAS 指标" in report
    assert "dry_run_fixture" in (run_dir / "ragas_results.jsonl").read_text(encoding="utf-8")
