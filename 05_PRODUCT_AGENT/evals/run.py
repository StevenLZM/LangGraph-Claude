from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    answer = str(response.get("answer", ""))
    expected_keywords = list(case.get("expected_keywords") or [])
    matched_keywords = [keyword for keyword in expected_keywords if keyword in answer]
    keyword_score = (
        round(len(matched_keywords) / len(expected_keywords) * 70)
        if expected_keywords
        else 70
    )

    expected_transfer = bool(case.get("expected_transfer", False))
    transfer_score = 15 if bool(response.get("needs_human_transfer", False)) == expected_transfer else 0

    expected_tool = str(case.get("expected_tool") or "")
    actual_tool = str((response.get("llm_trace") or {}).get("tool_name") or "")
    tool_score = 15 if not expected_tool or actual_tool == expected_tool else 0

    score = min(100, keyword_score + transfer_score + tool_score)
    return {
        "id": case["id"],
        "category": case["category"],
        "score": score,
        "passed": score >= 80,
        "matched_keywords": matched_keywords,
        "expected_keywords": expected_keywords,
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "expected_transfer": expected_transfer,
        "actual_transfer": bool(response.get("needs_human_transfer", False)),
        "answer": answer,
        "quality_score": response.get("quality_score"),
    }


def run_evaluation(
    *,
    dataset_path: Path,
    output_root: Path,
    chat_callable: Callable[[dict[str, Any]], dict[str, Any]],
    run_id: str | None = None,
) -> Path:
    from evals.report import build_report

    cases = load_dataset(dataset_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    with results_path.open("w", encoding="utf-8") as file:
        for case in cases:
            response = chat_callable(case)
            result = evaluate_case(case, response)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    (run_dir / "REPORT.md").write_text(build_report(results_path), encoding="utf-8")
    return run_dir


def _call_local_chat(case: dict[str, Any]) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "user_id": case.get("user_id", f"eval_{case['category']}"),
            "session_id": case.get("session_id", f"eval_{case['id']}"),
            "message": case["message"],
        },
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--output-root", default="evals/results")
    args = parser.parse_args()
    run_dir = run_evaluation(
        dataset_path=Path(args.dataset),
        output_root=Path(args.output_root),
        chat_callable=_call_local_chat,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
