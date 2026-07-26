from pathlib import Path
import json
import sys

import pytest
from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_cases(path: Path, cases: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )


def _valid_case(case_id: str, question: str) -> dict:
    return {
        "id": case_id,
        "category": "precise",
        "query_type": "keyword",
        "question": question,
        "reference": "产品保修期为 12 个月。",
        "expected_behavior": "answer",
        "auth_context": {},
        "qrels": [
            {
                "evidence_id": f"{case_id}-evidence",
                "source": "manual.pdf",
                "page_range": "3",
                "section": "保修政策",
                "evidence_text": "保修期为 12 个月",
                "evidence_hash": "pending-resolver-validation",
                "grade": 3,
            }
        ],
    }


def test_qrel_requires_grade_between_zero_and_three():
    from evals.models import EvidenceQrel

    with pytest.raises(ValueError, match="grade"):
        EvidenceQrel.from_dict(
            {
                "evidence_id": "e-1",
                "source": "manual.pdf",
                "page_range": "3",
                "section": "保修",
                "evidence_text": "保修期为 12 个月",
                "evidence_hash": "abc",
                "grade": 4,
            }
        )


def test_case_requires_reviewed_qrels_for_answer_behavior():
    from evals.models import EvaluationCase

    with pytest.raises(ValueError, match="qrels"):
        EvaluationCase.from_dict(
            {
                "id": "case-1",
                "category": "precise",
                "query_type": "keyword",
                "question": "保修多久",
                "reference": "12 个月",
                "expected_behavior": "answer",
                "auth_context": {},
                "qrels": [],
            },
            split="test",
        )


def test_validate_dataset_rejects_same_question_in_train_and_test(tmp_path):
    from evals.models import validate_dataset

    _write_cases(tmp_path / "train.jsonl", [_valid_case("train-1", "产品保修多久？")])
    _write_cases(tmp_path / "test.jsonl", [_valid_case("test-1", " 产品保修多久? ")])
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="跨 split"):
        validate_dataset(tmp_path, release_mode=False)


def test_release_mode_requires_two_hundred_test_cases(tmp_path):
    from evals.models import validate_dataset

    _write_cases(
        tmp_path / "test.jsonl",
        [_valid_case(f"test-{index}", f"问题 {index}") for index in range(199)],
    )
    (tmp_path / "train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "dev.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="至少 200"):
        validate_dataset(tmp_path, release_mode=True)


def test_case_split_comes_from_physical_file(tmp_path):
    from evals.models import load_split

    case = _valid_case("dev-1", "保修多久？")
    case["split"] = "test"
    _write_cases(tmp_path / "dev.jsonl", [case])

    loaded = load_split(tmp_path, "dev")

    assert loaded[0].split == "dev"


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

    row = build_ragas_row(
        case,
        docs,
        {"answer": "产品保修期为 12 个月。", "sources": docs},
    )

    assert row["user_input"] == "产品保修期多久？"
    assert row["retrieved_contexts"] == ["产品保修期为 12 个月，从购买之日起计算。"]
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


def test_answer_relevancy_metric_uses_single_generation(monkeypatch):
    from evals import ragas_adapter

    class FakeMetric:
        def __init__(self):
            self.strictness = 3

    monkeypatch.setattr(
        ragas_adapter,
        "_import_ragas_metric_factories",
        lambda: {
            name: FakeMetric
            for name in ragas_adapter.RAGAS_METRIC_NAMES
        },
    )

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

    assert _build_ragas_run_config().max_workers == 1
