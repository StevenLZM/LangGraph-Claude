from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.ragas_adapter import (  # noqa: E402
    RAGAS_METRIC_NAMES,
    RagasEvaluator,
    build_ragas_row,
    run_ragas_evaluation,
)


RetrievalCallable = Callable[[dict[str, Any]], list[Any]]
GenerationCallable = Callable[[dict[str, Any], list[Any]], dict[str, Any]]


REQUIRED_FIELDS = {"id", "category", "question", "reference"}


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        _validate_case(case, line_number)
        cases.append(case)
    return cases


def run_evaluation(
    *,
    dataset_path: Path,
    output_root: Path,
    retrieval_callable: RetrievalCallable | None = None,
    generation_callable: GenerationCallable | None = None,
    ragas_evaluator: RagasEvaluator | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    metric_names: list[str] | None = None,
) -> Path:
    from evals.report import build_report

    cases = load_dataset(dataset_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    retrieval = retrieval_callable or (_call_dry_run_retrieval if dry_run else _call_local_retrieval)
    generation = generation_callable or (_call_dry_run_generation if dry_run else _call_local_generation)

    ragas_rows: list[dict[str, Any]] = []
    for case in cases:
        docs = retrieval(case)
        response = generation(case, docs)
        ragas_rows.append(build_ragas_row(case, docs, response))

    metrics = metric_names or RAGAS_METRIC_NAMES
    effective_evaluator = ragas_evaluator
    if dry_run and effective_evaluator is None:
        effective_evaluator = _dry_run_ragas_evaluator
    results = run_ragas_evaluation(
        ragas_rows,
        metric_names=metrics,
        evaluator=effective_evaluator,
    )

    results_path = run_dir / "ragas_results.jsonl"
    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = build_report(
        results_path,
        metric_names=metrics,
        summary_path=run_dir / "summary.json",
    )
    (run_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return run_dir


def _validate_case(case: dict[str, Any], line_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - set(case))
    if missing:
        raise ValueError(f"dataset line {line_number} missing required fields: {missing}")
    if not str(case.get("reference") or "").strip():
        raise ValueError(f"dataset line {line_number} must include a non-empty reference")


def _call_local_retrieval(case: dict[str, Any]) -> list[Any]:
    from rag.query_rewriter import rewrite_query
    from rag.retriever import retrieve_with_hybrid

    rewrite = rewrite_query(case["question"], use_llm=False)
    case["rewritten_query"] = rewrite["rewritten_query"]
    case["actual_time_intent"] = rewrite["time_intent"]
    return retrieve_with_hybrid(
        query=rewrite["rewritten_query"],
        time_intent=rewrite["time_intent"],
    )


def _call_dry_run_retrieval(case: dict[str, Any]) -> list[Any]:
    parent_ids = list(case.get("expected_parent_ids") or [f"dry_parent_{case['id']}"])
    sources = list(case.get("expected_sources") or ["dry_source.pdf"])
    sections = list(case.get("expected_sections") or ["dry section"])
    reference = str(case.get("reference") or case["question"])
    metadata: dict[str, Any] = {
        "parent_id": parent_ids[0],
        "source": sources[0],
        "section_path": sections[0],
    }
    return [Document(page_content=reference, metadata=metadata)]


def _call_local_generation(case: dict[str, Any], docs: list[Any]) -> dict[str, Any]:
    from rag.chain import create_chain_with_history

    chain, _ = create_chain_with_history()
    result = chain.invoke(
        {"question": case["question"], "chat_history": []},
        config={"configurable": {"session_id": f"eval_{case['id']}"}},
    )
    result.setdefault("sources", docs)
    return result


def _call_dry_run_generation(case: dict[str, Any], docs: list[Any]) -> dict[str, Any]:
    return {
        "answer": str(case.get("reference") or "dry-run answer"),
        "sources": docs,
    }


def _dry_run_ragas_evaluator(
    rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            **{metric_name: 1.0 for metric_name in metric_names},
            "ragas_mode": "dry_run_fixture",
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 01_RAG RAGAS evals.")
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--output-root", default="evals/results")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = run_evaluation(
        dataset_path=Path(args.dataset),
        output_root=Path(args.output_root),
        dry_run=args.dry_run,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
