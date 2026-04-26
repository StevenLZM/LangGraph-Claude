"""evals/run.py —— 跑数据集，调真 graph + judge，落 results.jsonl + REPORT.md。

用法：
    PYTHONPATH=. python -m evals.run                  # 全量（5 题）
    PYTHONPATH=. python -m evals.run --limit 1        # 烟测 1 题
    PYTHONPATH=. python -m evals.run --dataset xxx.jsonl

产出目录：evals/results/{ts}/
    results.jsonl   每行一个 case 的输入 / 报告 / 打分 / 耗时
    REPORT.md       由 evals.report 渲染的可读评测报告
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from langgraph.types import Command

from app import bootstrap
from evals.judge import JudgeInput, judge_one
from evals.report import render_markdown

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset.jsonl"
RESULTS_ROOT = ROOT / "results"


def _ts_slug() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def _evidence_brief(evidence: list[Any]) -> list[dict]:
    out = []
    for ev in evidence or []:
        if hasattr(ev, "model_dump"):
            d = ev.model_dump()
        elif isinstance(ev, dict):
            d = dict(ev)
        else:
            continue
        out.append({
            "source_type": d.get("source_type"),
            "source_url": d.get("source_url"),
            "snippet": d.get("snippet", ""),
        })
    return out


def _plan_brief(plan: list[Any]) -> list[dict]:
    out = []
    for sq in plan or []:
        if hasattr(sq, "model_dump"):
            d = sq.model_dump()
        elif isinstance(sq, dict):
            d = dict(sq)
        else:
            continue
        out.append({
            "id": d.get("id"),
            "question": d.get("question"),
            "recommended_sources": d.get("recommended_sources", []),
        })
    return out


async def _run_one(case: dict, run_id: str) -> dict:
    g = bootstrap.app_state.graph
    tid = f"eval-{run_id}-{case['id']}"
    cfg = {
        "configurable": {"thread_id": tid},
        "metadata": {
            "eval_run_id": run_id,
            "case_id": case["id"],
            "category": case.get("category"),
            "audience": case.get("audience", "intermediate"),
            "research_query": case["query"],
            "app": "insightloop-eval",
        },
        "tags": ["eval", f"run:{run_id}", f"case:{case['id']}"],
    }
    started = time.time()
    record: dict[str, Any] = {"case": case, "thread_id": tid, "error": None}
    try:
        r1 = await g.ainvoke(
            {
                "research_query": case["query"],
                "audience": case.get("audience", "intermediate"),
                "messages": [],
                "evidence": [],
            },
            config=cfg,
        )
        intr = r1.get("__interrupt__")
        if not intr:
            raise RuntimeError("planner 未触发 interrupt")
        proposed = intr[0].value if isinstance(intr, list) else intr.value
        plan_payload = proposed.get("plan") or {}

        r2 = await g.ainvoke(Command(resume={"plan": plan_payload}), config=cfg)
        plan = r2.get("plan") or []
        evidence = r2.get("evidence") or []
        report_md = r2.get("final_report", "") or ""
        record.update({
            "report_path": r2.get("report_path"),
            "report_md": report_md,
            "evidence_count": len(evidence),
            "plan_size": len(plan),
        })

        score = await judge_one(JudgeInput(
            query=case["query"],
            plan=_plan_brief(plan),
            evidence_brief=_evidence_brief(evidence),
            report_md=report_md,
        ))
        record["score"] = score.model_dump()
    except Exception as e:
        logger.exception("[eval] case %s 失败: %s", case["id"], e)
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        record["elapsed_sec"] = round(time.time() - started, 1)
    return record


async def main_async(args: argparse.Namespace) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dataset_path = Path(args.dataset)
    cases = load_dataset(dataset_path)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("数据集为空")

    run_id = _ts_slug()
    out_dir = RESULTS_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    await bootstrap.startup()
    try:
        with results_path.open("w", encoding="utf-8") as f:
            for i, case in enumerate(cases, 1):
                logger.info("[eval] (%d/%d) running case %s", i, len(cases), case["id"])
                rec = await _run_one(case, run_id)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        await bootstrap.shutdown()

    report_path = render_markdown(results_path, out_dir / "REPORT.md", run_id=run_id)
    logger.info("[eval] 完成 run_id=%s → %s", run_id, report_path)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="InsightLoop 评测脚本")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0, help="0=全部")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
