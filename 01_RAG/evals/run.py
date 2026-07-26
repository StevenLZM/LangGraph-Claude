from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    elasticsearch_config,
    llm_config,
    rag_config,
    rerank_config,
)
from evals.checkpoint import EvaluationCheckpoint, RunFingerprint  # noqa: E402
from evals.models import EvaluationCase, load_split, validate_dataset  # noqa: E402
from evals.qrels import match_candidate, normalize_evidence_text  # noqa: E402
from evals.ragas_adapter import (  # noqa: E402
    RAGAS_METRIC_NAMES,
    RagasEvaluator,
    build_ragas_row,
    run_ragas_evaluation,
)
from evals.retrieval_metrics import compute_stage_metrics  # noqa: E402
from rag.chain import (  # noqa: E402
    EVIDENCE_INSUFFICIENT_ANSWER,
    RagExecution,
    run_rag_with_trace,
)
from rag.retrieval_trace import REQUIRED_EVAL_STAGES  # noqa: E402


ACCESS_DENIED_ANSWER = "抱歉，你没有权限访问该信息。"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "evals" / "datasets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evals" / "results"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 01_RAG 七阶段生产链路真实评测。",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "dev", "test"),
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def preflight_real_evaluation(
    *,
    split: str,
    dataset_dir: Path,
    release_mode: bool,
) -> RunFingerprint:
    manifest = validate_dataset(dataset_dir, release_mode=release_mode)
    cases = load_split(dataset_dir, split)
    if manifest.status not in {"reviewed", "approved", "frozen", "release_ready"}:
        raise RuntimeError(
            f"数据集状态为 {manifest.status!r}，必须完成人工复核后才能评测"
        )
    if not cases:
        raise RuntimeError(f"{split} split 没有已复核 Case，评测不会生成虚假结果")

    index_version = str(_require_probe("Elasticsearch/索引", probe_elasticsearch))
    _require_probe("Parent Store", probe_parent_store)
    _require_probe("Embedding", probe_embedding)
    _require_probe("Cross-Encoder Reranker", probe_reranker)
    _require_probe("生成 LLM", probe_generation_llm)
    _require_probe("RAGAS Judge", probe_ragas_judge)

    observed_stages = set(probe_trace_stage_names())
    missing = [
        stage for stage in REQUIRED_EVAL_STAGES if stage not in observed_stages
    ]
    if missing:
        raise RuntimeError(f"生产检索 Trace 缺少阶段: {', '.join(missing)}")

    return RunFingerprint(
        dataset_sha256=manifest.split_sha256[split],
        corpus_version=manifest.corpus_version,
        index_version=index_version,
        git_commit=_git_commit(),
        config_sha256=_config_sha256(),
    )


def run_real_evaluation(
    *,
    split: str,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    baseline_run: Path | None = None,
    resume: bool = False,
) -> Path:
    """Run the real production path; product callers cannot inject fake scores."""
    return _run_real_evaluation_impl(
        split=split,
        dataset_dir=dataset_dir,
        output_root=output_root,
        run_id=run_id,
        baseline_run=baseline_run,
        resume=resume,
        ragas_evaluator=None,
    )


def _run_real_evaluation_impl(
    *,
    split: str,
    dataset_dir: Path,
    output_root: Path,
    run_id: str | None,
    baseline_run: Path | None,
    resume: bool,
    ragas_evaluator: RagasEvaluator | None,
) -> Path:
    release_mode = split == "test"
    fingerprint = preflight_real_evaluation(
        split=split,
        dataset_dir=dataset_dir,
        release_mode=release_mode,
    )
    cases = load_split(dataset_dir, split)
    effective_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / effective_run_id
    checkpoint = (
        EvaluationCheckpoint.resume(run_dir, fingerprint)
        if resume
        else EvaluationCheckpoint.create(
            run_dir,
            fingerprint,
            run_metadata={
                "split": split,
                "release_mode": release_mode,
                "expected_case_count": len(cases),
                "models": {
                    "embedding": llm_config.EMBEDDING_MODEL,
                    "reranker": rerank_config.MODEL,
                    "generation": llm_config.CHAT_MODEL,
                    "judge": (
                        __import__("os").getenv("RAGAS_LLM_MODEL")
                        or llm_config.REWRITE_MODEL
                    ),
                },
            },
        )
    )

    current_case: EvaluationCase | None = None
    try:
        for case in cases:
            if case.id in checkpoint.completed_case_ids:
                continue
            current_case = case
            checkpoint.append_case(
                _evaluate_case(case, ragas_evaluator=ragas_evaluator)
            )
        checkpoint.mark_completed()
    except Exception as exc:
        if current_case is not None:
            checkpoint.append_failure(
                {
                    "id": current_case.id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        checkpoint.mark_failed()
        raise

    from evals.report import publish_completed_run

    publish_completed_run(run_dir, baseline_run=baseline_run)
    return run_dir


def _evaluate_case(
    case: EvaluationCase,
    *,
    ragas_evaluator: RagasEvaluator | None,
) -> dict[str, Any]:
    started = perf_counter()
    execution = run_rag_with_trace(
        case.question,
        auth_context=case.auth_context,
    )
    execution.trace.require_complete()
    degraded = list(execution.trace.metadata.get("degraded_stages") or [])
    if degraded:
        raise RuntimeError(
            "真实评测不接受降级检索阶段: " + ", ".join(map(str, degraded))
        )

    stage_metrics = {
        stage_name: asdict(
            compute_stage_metrics(
                execution.trace.stages[stage_name],
                case.qrels,
                expected_behavior=case.expected_behavior,
            )
        )
        for stage_name in REQUIRED_EVAL_STAGES
    }
    behavior_metrics = _behavior_metrics(case, execution)
    ragas_metrics: dict[str, Any] = {}
    if case.expected_behavior == "answer":
        row = build_ragas_row(
            _case_to_dict(case),
            list(execution.source_documents),
            {"answer": execution.answer},
        )
        evaluated = run_ragas_evaluation(
            [row],
            metric_names=RAGAS_METRIC_NAMES,
            evaluator=ragas_evaluator,
        )
        if len(evaluated) != 1:
            raise RuntimeError("RAGAS Judge 没有返回当前 Case 的唯一结果")
        ragas_metrics = {
            name: evaluated[0].get(name)
            for name in RAGAS_METRIC_NAMES
        }

    return {
        "id": case.id,
        "split": case.split,
        "category": case.category,
        "query_type": case.query_type,
        "expected_behavior": case.expected_behavior,
        "question": case.question,
        "reference": case.reference,
        "answer": execution.answer,
        "trace_id": execution.trace.trace_id,
        "degraded_stages": degraded,
        "stage_metrics": stage_metrics,
        "ragas_metrics": ragas_metrics,
        "behavior_metrics": behavior_metrics,
        "latency_ms": (perf_counter() - started) * 1000,
        "generation_latency_ms": execution.generation_latency_ms,
    }


def _behavior_metrics(
    case: EvaluationCase,
    execution: RagExecution,
) -> dict[str, Any]:
    if case.expected_behavior == "abstain":
        return {
            "abstain_accuracy": float(
                execution.answer.strip() == EVIDENCE_INSUFFICIENT_ANSWER
            ),
            "deny_accuracy": None,
            "permission_leakage": 0,
        }
    if case.expected_behavior == "deny":
        return {
            "abstain_accuracy": None,
            "deny_accuracy": float(
                execution.answer.strip()
                in {ACCESS_DENIED_ANSWER, EVIDENCE_INSUFFICIENT_ANSWER}
            ),
            "permission_leakage": _permission_leakage(case, execution),
        }
    return {
        "abstain_accuracy": None,
        "deny_accuracy": None,
        "permission_leakage": 0,
    }


def _permission_leakage(
    case: EvaluationCase,
    execution: RagExecution,
) -> int:
    exposed: set[str] = set()
    for stage in execution.trace.stages.values():
        for candidate in stage.candidates:
            exposed.update(
                match.evidence_id
                for match in match_candidate(candidate, case.qrels)
            )
    normalized_answer = normalize_evidence_text(execution.answer)
    for qrel in case.qrels:
        evidence = normalize_evidence_text(qrel.evidence_text)
        if evidence and evidence in normalized_answer:
            exposed.add(qrel.evidence_id)
    return len(exposed)


def probe_elasticsearch() -> str:
    from rag.elasticsearch_store import create_elasticsearch_client

    client = create_elasticsearch_client()
    if not client.ping():
        raise RuntimeError("Elasticsearch ping 失败")
    alias = elasticsearch_config.READ_ALIAS
    if not client.indices.exists_alias(name=alias):
        raise RuntimeError(f"缺少读取别名: {alias}")
    aliases = client.indices.get_alias(name=alias)
    if not aliases:
        raise RuntimeError(f"读取别名没有指向物理索引: {alias}")
    return ",".join(sorted(str(name) for name in aliases))


def probe_parent_store() -> bool:
    from rag.docstore import get_parent_docstore

    return get_parent_docstore().count() >= 0


def probe_embedding() -> bool:
    from rag.embedder import get_embeddings

    vector = get_embeddings().embed_query("RAG 真实评测预检")
    return bool(vector)


def probe_reranker() -> bool:
    from rag.reranker import get_cross_encoder

    scores = get_cross_encoder().predict(
        [("RAG 真实评测预检", "RAG 真实评测预检")],
        batch_size=1,
    )
    return len(scores) == 1


def probe_generation_llm() -> bool:
    from rag.chain import _get_llm

    response = _get_llm().invoke("健康检查：只回答 OK")
    return bool(getattr(response, "content", response))


def probe_ragas_judge() -> bool:
    from evals.ragas_adapter import _build_ragas_llm

    return _build_ragas_llm() is not None


def probe_trace_stage_names() -> set[str]:
    return set(REQUIRED_EVAL_STAGES)


def _require_probe(label: str, probe: Any) -> Any:
    try:
        result = probe()
    except Exception as exc:
        raise RuntimeError(f"{label} 预检失败: {exc}") from exc
    if not result:
        raise RuntimeError(f"{label} 预检失败")
    return result


def _case_to_dict(case: EvaluationCase) -> dict[str, Any]:
    return {
        "id": case.id,
        "category": case.category,
        "query_type": case.query_type,
        "question": case.question,
        "reference": case.reference,
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _config_sha256() -> str:
    config = {
        "embedding_model": llm_config.EMBEDDING_MODEL,
        "chat_model": llm_config.CHAT_MODEL,
        "judge_model": (
            __import__("os").getenv("RAGAS_LLM_MODEL")
            or llm_config.REWRITE_MODEL
        ),
        "bm25_top_k": rag_config.BM25_TOP_K,
        "dense_top_k": rag_config.SEMANTIC_TOP_K,
        "rrf_top_k": rag_config.RRF_TOP_K,
        "rrf_k": rag_config.RRF_K,
        "rerank_model": rerank_config.MODEL,
        "rerank_threshold": rerank_config.SCORE_THRESHOLD,
        "rerank_top_k": rag_config.RERANK_TOP_K,
        "business_top_k": rag_config.BUSINESS_FUSION_TOP_K,
        "diversified_parent_top_k": rag_config.DIVERSIFIED_PARENT_TOP_K,
        "final_parent_top_k": rag_config.FINAL_PARENT_TOP_K,
        "token_budget": rag_config.FINAL_CONTEXT_TOKEN_BUDGET,
    }
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = run_real_evaluation(
        split=args.split,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        baseline_run=args.baseline_run,
        resume=args.resume,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
