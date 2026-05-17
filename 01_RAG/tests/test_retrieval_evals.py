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


def test_retrieval_ir_metrics_compute_recall_mrr_and_hit_at_k():
    from evals.retrieval_metrics import compute_retrieval_metrics

    metrics = compute_retrieval_metrics(
        expected_ids=["manual.pdf", "faq.pdf"],
        retrieved_ids=["other.pdf", "manual.pdf", "manual.pdf", "guide.pdf"],
        k=3,
    )

    assert metrics == {
        "retrieval_recall_at_3": 0.5,
        "retrieval_mrr": 0.5,
        "retrieval_hit_at_3": 1.0,
    }


def test_answer_relevancy_metric_uses_single_generation_for_openai_compatible_llms(monkeypatch):
    from evals import ragas_adapter

    class FakeMetric:
        def __init__(self):
            self.strictness = 3

    def fake_import_metrics():
        return {
            "context_precision": FakeMetric,
            "context_recall": FakeMetric,
            "faithfulness": FakeMetric,
            "answer_correctness": FakeMetric,
            "answer_relevancy": FakeMetric,
            "semantic_similarity": FakeMetric,
        }

    monkeypatch.setattr(ragas_adapter, "_import_ragas_metric_factories", fake_import_metrics)

    metric = ragas_adapter._load_ragas_metrics(["answer_relevancy"])[0]

    assert metric.strictness == 1


def test_ragas_llm_uses_rewrite_model_by_default(monkeypatch):
    from evals import ragas_adapter

    captured = {}

    class FakeWrapper:
        def __init__(self, llm):
            self.llm = llm

    def fake_get_llm(model_name=None):
        captured["model_name"] = model_name
        return {"model_name": model_name}

    monkeypatch.setattr(ragas_adapter, "LangchainLLMWrapper", FakeWrapper)
    monkeypatch.setattr(ragas_adapter, "_get_llm", fake_get_llm)
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)

    ragas_adapter._build_ragas_llm()

    assert captured["model_name"] == ragas_adapter.llm_config.REWRITE_MODEL


def test_ragas_run_config_defaults_to_single_worker(monkeypatch):
    from evals.ragas_adapter import _build_ragas_run_config

    monkeypatch.delenv("RAGAS_MAX_WORKERS", raising=False)

    run_config = _build_ragas_run_config()

    assert run_config.max_workers == 1


def test_report_summary_ignores_nan_and_infinite_metric_values():
    from evals.report import summarize_results

    summary = summarize_results(
        [
            {"category": "precise", "context_precision": 1.0},
            {"category": "precise", "context_precision": float("nan")},
            {"category": "precise", "context_precision": float("inf")},
        ],
        ["context_precision"],
    )

    assert summary["metrics"]["context_precision"] == {"average": 1.0, "count": 1}


def test_run_ragas_evaluation_writes_results_summary_and_report(tmp_path):
    from evals.run import run_evaluation

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"reference":"产品保修期为 12 个月。",'
        '"expected_sources":["spec.pdf","faq.pdf"]}\n',
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
    assert result["retrieval_recall_at_5"] == 0.5
    assert result["retrieval_mrr"] == 1.0
    assert result["retrieval_hit_at_5"] == 1.0
    assert result["retrieval_eval_target"] == "source"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["metrics"]["context_precision"]["average"] == 1.0
    assert summary["metrics"]["faithfulness"]["average"] == 0.95
    assert summary["metrics"]["answer_correctness"]["average"] == 0.93
    assert summary["retrieval_metrics"]["retrieval_recall_at_5"]["average"] == 0.5
    assert summary["retrieval_metrics"]["retrieval_mrr"]["average"] == 1.0
    assert summary["retrieval_metrics"]["retrieval_hit_at_5"]["average"] == 1.0

    report = report_path.read_text(encoding="utf-8")
    assert "# 01_RAG 离线评估报告" in report
    assert "Context Precision" in report
    assert "Answer Correctness" in report
    assert "Semantic Similarity" in report
    assert "传统 IR 检索指标" in report
    assert "Recall@5" in report
    assert "MRR" in report
    assert "Hit@5" in report
    assert "关键词" not in report


def test_run_evaluation_respects_custom_metric_names(tmp_path):
    from evals.run import run_evaluation

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"reference":"产品保修期为 12 个月。"}\n',
        encoding="utf-8",
    )

    def fake_retriever(case):
        return [Document(page_content="产品保修期为 12 个月。", metadata={})]

    def fake_generator(case, docs):
        return {"answer": "产品保修期为 12 个月。", "sources": docs}

    def fake_ragas_evaluator(rows, metric_names):
        assert metric_names == ["context_precision", "context_recall"]
        return [{**rows[0], "context_precision": 1.0, "context_recall": 1.0}]

    run_dir = run_evaluation(
        dataset_path=dataset_path,
        output_root=tmp_path / "results",
        retrieval_callable=fake_retriever,
        generation_callable=fake_generator,
        ragas_evaluator=fake_ragas_evaluator,
        run_id="custom-metrics",
        metric_names=["context_precision", "context_recall"],
    )

    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "Context Precision" in report
    assert "Context Recall" in report
    assert "Answer Relevancy" not in report


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
    assert "01_RAG 离线评估报告" in report
    assert "RAGAS 指标" in report
    assert "dry_run_fixture" in (run_dir / "ragas_results.jsonl").read_text(encoding="utf-8")
