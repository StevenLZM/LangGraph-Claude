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

from evals.metrics import evaluate_generation_case, evaluate_retrieval_case


RetrievalCallable = Callable[[dict[str, Any]], list[Any]]
GenerationCallable = Callable[[dict[str, Any], list[Any]], dict[str, Any]]


REQUIRED_FIELDS = {"id", "category", "question"}
GOLD_FIELD_PREFIXES = ("expected_", "answer_", "forbidden_")


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
    run_id: str | None = None,
    with_generation: bool = False,
    with_judge: bool = False,
    dry_run: bool = False,
) -> Path:
    from evals.judge import judge_generation
    from evals.report import build_report

    cases = load_dataset(dataset_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    retrieval = retrieval_callable or (_call_dry_run_retrieval if dry_run else _call_local_retrieval)
    generation = generation_callable or (_call_dry_run_generation if dry_run else _call_local_generation)

    with results_path.open("w", encoding="utf-8") as file:
        for case in cases:
            docs = retrieval(case)
            result = evaluate_retrieval_case(case, docs, k_values=(3, 5))
            if with_generation:
                response = generation(case, docs)
                result.update(evaluate_generation_case(case, response))
                if with_judge:
                    result.update(judge_generation(case, response))
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    build_report(results_path, summary_path=run_dir / "summary.json")
    (run_dir / "REPORT.md").write_text(build_report(results_path), encoding="utf-8")
    return run_dir


def _validate_case(case: dict[str, Any], line_number: int) -> None:
    missing = sorted(REQUIRED_FIELDS - set(case))
    if missing:
        raise ValueError(f"dataset line {line_number} missing required fields: {missing}")
    has_gold = any(
        key.startswith(GOLD_FIELD_PREFIXES)
        for key in case
    )
    if not has_gold:
        raise ValueError(
            f"dataset line {line_number} must include at least one expected_ or answer_ gold field"
        )


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
    keywords = list(case.get("expected_keywords") or [])
    expected_time = case.get("expected_time_range") or {}
    metadata: dict[str, Any] = {
        "parent_id": parent_ids[0],
        "source": sources[0],
        "section_path": sections[0],
    }
    if expected_time:
        field = expected_time.get("field", "doc_date")
        if field == "upload_date":
            metadata["upload_date"] = int(expected_time.get("gte") or expected_time.get("lte") or 0)
        else:
            metadata.update(
                {
                    "has_doc_date": True,
                    "doc_date_min": int(expected_time.get("gte") or 0),
                    "doc_date_max": int(expected_time.get("lte") or 99991231),
                }
            )
    return [Document(page_content=" ".join(keywords) or case["question"], metadata=metadata)]


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
    keywords = list(case.get("answer_keywords") or case.get("expected_keywords") or [])
    return {
        "answer": " ".join(keywords) or "dry-run answer",
        "sources": docs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 01_RAG retrieval and generation evals.")
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--output-root", default="evals/results")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = run_evaluation(
        dataset_path=Path(args.dataset),
        output_root=Path(args.output_root),
        with_generation=args.with_generation,
        with_judge=args.with_judge,
        dry_run=args.dry_run,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
