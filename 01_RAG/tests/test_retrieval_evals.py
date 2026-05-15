from pathlib import Path
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_retrieval_metrics_rank_and_parent_hits():
    from evals.metrics import evaluate_retrieval_case

    case = {
        "id": "warranty_001",
        "category": "precise",
        "question": "产品保修期多久？",
        "expected_parent_ids": ["parent-warranty"],
        "expected_sources": ["spec.pdf"],
        "expected_sections": ["第一章 产品概述"],
        "expected_keywords": ["保修期", "12 个月"],
    }
    docs = [
        Document(
            page_content="系统要求 Python 3.9+。",
            metadata={
                "parent_id": "parent-tech",
                "source": "spec.pdf",
                "section_path": "第二章 技术规格",
            },
        ),
        Document(
            page_content="产品保修期为 12 个月，从购买之日起计算。",
            metadata={
                "parent_id": "parent-warranty",
                "source": "spec.pdf",
                "section_path": "第一章 产品概述",
            },
        ),
    ]

    result = evaluate_retrieval_case(case, docs, k_values=[1, 2])

    assert result["recall@1"] == 0.0
    assert result["recall@2"] == 1.0
    assert result["mrr"] == 0.5
    assert result["parent_hit"] is True
    assert result["source_hit"] is True
    assert result["context_completeness"] == 1.0


def test_retrieval_metrics_time_filter_accuracy():
    from evals.metrics import evaluate_retrieval_case

    case = {
        "id": "invoice_2024",
        "category": "time",
        "question": "2024 年的发票",
        "expected_parent_ids": ["invoice-2024"],
        "expected_time_range": {"field": "doc_date", "gte": 20240101, "lte": 20241231},
    }
    docs = [
        Document(
            page_content="2024 年珠海发票",
            metadata={
                "parent_id": "invoice-2024",
                "source": "珠海发票.pdf",
                "has_doc_date": True,
                "doc_date_min": 20240401,
                "doc_date_max": 20240401,
            },
        ),
        Document(
            page_content="2023 年新疆发票",
            metadata={
                "parent_id": "invoice-2023",
                "source": "新疆发票.pdf",
                "has_doc_date": True,
                "doc_date_min": 20231201,
                "doc_date_max": 20231201,
            },
        ),
    ]

    result = evaluate_retrieval_case(case, docs, k_values=[2])

    assert result["time_filter_accuracy"] == 0.5


def test_retrieval_metrics_time_intent_accuracy():
    from evals.metrics import evaluate_retrieval_case

    case = {
        "id": "invoice_2024",
        "category": "time",
        "question": "2024 年的发票",
        "expected_sources": ["珠海发票.pdf"],
        "expected_time_intent": {
            "type": "year",
            "field": "doc_date",
            "range": {"gte": 20240101, "lte": 20241231},
            "sort": None,
        },
        "actual_time_intent": {
            "type": "year",
            "field": "doc_date",
            "range": {"gte": 20240101, "lte": 20241231},
            "sort": None,
        },
    }

    result = evaluate_retrieval_case(case, [], k_values=[1])

    assert result["time_intent_accuracy"] == 1.0


def test_generation_metrics_use_keywords_sources_and_forbidden_terms():
    from evals.metrics import evaluate_generation_case

    case = {
        "id": "warranty_answer",
        "category": "precise",
        "answer_keywords": ["保修期", "12 个月"],
        "forbidden_keywords": ["24 个月"],
        "expected_sources": ["spec.pdf"],
    }
    response = {
        "answer": "产品保修期为 12 个月。[来源: spec.pdf, 第1页]",
        "sources": [
            Document(page_content="产品保修期为 12 个月。", metadata={"source": "spec.pdf"})
        ],
    }

    result = evaluate_generation_case(case, response)

    assert result["answer_keyword_coverage"] == 1.0
    assert result["forbidden_keyword_hits"] == []
    assert result["citation_source_accuracy"] == 1.0
    assert result["generation_score"] == 100


def test_dataset_loader_validates_required_fields(tmp_path):
    from evals.run import load_dataset

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"expected_sources":["spec.pdf"],"expected_keywords":["保修期"]}\n',
        encoding="utf-8",
    )

    cases = load_dataset(dataset_path)

    assert cases[0]["id"] == "case_001"


def test_dataset_loader_rejects_missing_gold_fields(tmp_path):
    import pytest

    from evals.run import load_dataset

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_"):
        load_dataset(dataset_path)


def test_run_retrieval_evaluation_writes_results_and_report(tmp_path):
    from evals.run import run_evaluation

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"case_001","category":"precise","question":"保修期多久？",'
        '"expected_parent_ids":["parent-warranty"],"expected_sources":["spec.pdf"],'
        '"expected_keywords":["保修期","12 个月"],"answer_keywords":["保修期","12 个月"]}\n',
        encoding="utf-8",
    )

    def fake_retriever(case):
        assert case["question"] == "保修期多久？"
        return [
            Document(
                page_content="产品保修期为 12 个月。",
                metadata={"parent_id": "parent-warranty", "source": "spec.pdf"},
            )
        ]

    run_dir = run_evaluation(
        dataset_path=dataset_path,
        output_root=tmp_path / "results",
        retrieval_callable=fake_retriever,
        run_id="unit-run",
        with_generation=False,
    )

    assert (run_dir / "results.jsonl").exists()
    report = (run_dir / "REPORT.md").read_text(encoding="utf-8")
    assert "# 01_RAG 检索效果评测报告" in report
    assert "Recall@5" in report
    assert "precise" in report


def test_packaged_eval_dataset_is_valid():
    from evals.run import load_dataset

    dataset_path = Path(__file__).parent.parent / "evals" / "dataset.jsonl"

    cases = load_dataset(dataset_path)

    assert len(cases) >= 5
    assert {case["category"] for case in cases} >= {"conceptual", "precise", "time"}


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
    assert "01_RAG 检索效果评测报告" in report
