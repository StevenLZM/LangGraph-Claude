from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from evals.report import build_report
from evals.run import _chat_payload_for_case, evaluate_case, load_dataset, run_evaluation


def test_m6_dataset_contains_100_balanced_customer_service_cases():
    cases = load_dataset(Path("evals/dataset.jsonl"))

    assert len(cases) == 100
    categories = {case["category"] for case in cases}
    assert {
        "order",
        "logistics",
        "product",
        "refund",
        "human_transfer",
        "memory",
        "faq_rag",
        "degraded",
    }.issubset(categories)
    assert all(case["id"] and case["message"] and case["expected_keywords"] for case in cases)


def test_evaluate_case_scores_keywords_transfer_and_rag_metadata():
    case = {
        "id": "faq_001",
        "category": "faq_rag",
        "message": "退货政策是什么？",
        "expected_keywords": ["7 天", "退货"],
        "expected_transfer": False,
        "expected_tool": "faq_rag",
    }
    response = {
        "answer": "签收 7 天内可申请退货。",
        "needs_human_transfer": False,
        "llm_trace": {"tool_name": "faq_rag"},
    }

    result = evaluate_case(case, response)

    assert result["score"] == 100
    assert result["passed"] is True
    assert result["matched_keywords"] == ["7 天", "退货"]


def test_run_evaluation_writes_results_and_markdown_report(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "order_001",
                        "category": "order",
                        "user_id": "eval_user",
                        "session_id": "eval_session",
                        "message": "我的订单 ORD123456 到哪了？",
                        "expected_keywords": ["ORD123456", "派送"],
                        "expected_transfer": False,
                        "expected_tool": "get_logistics",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "results"

    run_dir = run_evaluation(
        dataset_path=dataset_path,
        output_root=output_dir,
        chat_callable=lambda case: {
            "answer": "订单 ORD123456 正在派送。",
            "needs_human_transfer": False,
            "llm_trace": {"tool_name": "get_logistics"},
            "quality_score": 90,
        },
        run_id="test-run",
    )

    results_path = run_dir / "results.jsonl"
    report_path = run_dir / "REPORT.md"
    assert results_path.exists()
    assert report_path.exists()
    assert "平均分" in report_path.read_text(encoding="utf-8")

    report = build_report(results_path)
    assert "order" in report
    assert "100.0" in report


def test_local_chat_payload_includes_stable_request_id():
    case = {
        "id": "order_001",
        "category": "order",
        "user_id": "eval_user",
        "session_id": "eval_session",
        "message": "我的订单 ORD123456 到哪了？",
    }

    payload = _chat_payload_for_case(case)

    assert payload["request_id"] == "eval_order_001"
    assert payload["user_id"] == "eval_user"
    assert payload["session_id"] == "eval_session"
    assert payload["message"] == case["message"]


def test_eval_run_script_can_execute_from_project_root(tmp_path):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "id": "script_001",
                "category": "order",
                "user_id": "script_user",
                "session_id": "script_session",
                "message": "我的订单 ORD123456 到哪了？",
                "expected_keywords": ["ORD123456", "派送"],
                "expected_transfer": False,
                "expected_tool": "get_logistics",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            "evals/run.py",
            "--dataset",
            str(dataset_path),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_root / "REPORT.md").exists() or any(output_root.glob("*/REPORT.md"))
