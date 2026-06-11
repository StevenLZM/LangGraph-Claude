"""Trace metrics and manual-review persistence for production evals."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from evals.manual_review import (
    append_manual_review,
    build_review_queue,
    load_manual_reviews,
    write_review_queue,
)
from evals.trace_metrics import EvalTraceCollector


def test_eval_trace_collector_summarizes_chain_llm_tool_metrics():
    collector = EvalTraceCollector(case_id="tech_01")
    chain_id = uuid4()
    llm_id = uuid4()
    tool_id = uuid4()

    collector.on_chain_start(
        {"name": "planner"},
        {},
        run_id=chain_id,
        tags=["agent:planner"],
        metadata={"agent": "planner"},
    )
    collector.on_llm_start({"name": "ChatOpenAI"}, ["prompt"], run_id=llm_id, parent_run_id=chain_id)
    collector.on_llm_end(_fake_llm_result(prompt_tokens=10, completion_tokens=20), run_id=llm_id)
    collector.on_tool_start({"name": "web_search"}, "q", run_id=tool_id, parent_run_id=chain_id)
    collector.on_tool_error(RuntimeError("timeout"), run_id=tool_id)
    collector.on_chain_end({}, run_id=chain_id)

    summary = collector.summary()

    assert summary["case_id"] == "tech_01"
    assert summary["chain_runs"] == 1
    assert summary["llm_calls"] == 1
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 1
    assert summary["prompt_tokens"] == 10
    assert summary["completion_tokens"] == 20
    assert summary["total_tokens"] == 30
    assert summary["by_agent"]["planner"]["chain_runs"] == 1
    assert summary["by_agent"]["planner"]["llm_calls"] == 1
    assert summary["by_agent"]["planner"]["tool_errors"] == 1


def test_manual_review_queue_and_review_jsonl_round_trip(tmp_path: Path):
    records = [
        {
            "case": {"id": "tech_01", "query": "q1", "category": "技术"},
            "thread_id": "tid-1",
            "sampling": {
                "risk_level": "high",
                "review_action": "manual_review",
                "reasons": ["low_overall_score"],
            },
            "score": {"overall": 62},
            "report_path": "data/reports/a.md",
        },
        {
            "case": {"id": "tech_02", "query": "q2", "category": "技术"},
            "thread_id": "tid-2",
            "sampling": {
                "risk_level": "low",
                "review_action": "random_sample",
                "reasons": [],
            },
            "score": {"overall": 91},
        },
    ]

    queue = build_review_queue(records)

    assert len(queue) == 1
    assert queue[0]["case_id"] == "tech_01"
    assert queue[0]["status"] == "pending"
    assert queue[0]["suggested_label"] == "major_issue"

    queue_path = write_review_queue(queue, tmp_path / "manual_review_queue.jsonl")
    assert queue_path.read_text(encoding="utf-8").count("\n") == 1

    review_path = tmp_path / "manual_reviews.jsonl"
    append_manual_review(
        review_path,
        case_id="tech_01",
        reviewer="alice",
        label="major_issue",
        notes="引用不足",
        root_cause="research",
    )

    reviews = load_manual_reviews(review_path)
    assert reviews == [
        {
            "case_id": "tech_01",
            "reviewer": "alice",
            "label": "major_issue",
            "notes": "引用不足",
            "root_cause": "research",
        }
    ]


def test_run_one_records_callback_trace_metrics(monkeypatch):
    import asyncio

    from evals import run as eval_run

    class _Interrupt:
        value = {"plan": {"sub_questions": []}}

    class _FakeGraph:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            self.calls += 1
            callbacks = (config or {}).get("callbacks") or []
            for cb in callbacks:
                chain_id = uuid4()
                llm_id = uuid4()
                cb.on_chain_start(
                    {"name": "planner"},
                    {},
                    run_id=chain_id,
                    tags=["agent:planner"],
                    metadata={"agent": "planner"},
                )
                cb.on_llm_start({"name": "ChatOpenAI"}, ["prompt"], run_id=llm_id, parent_run_id=chain_id)
                cb.on_llm_end(_fake_llm_result(prompt_tokens=3, completion_tokens=4), run_id=llm_id)
                cb.on_chain_end({}, run_id=chain_id)
            if self.calls == 1:
                return {"__interrupt__": [_Interrupt()]}
            return {
                "plan": [],
                "evidence": [],
                "final_report": "# report",
                "report_path": "data/reports/x.md",
            }

    async def _fake_judge_one(_inp):
        class _Score:
            def model_dump(self):
                return {"coverage": 90, "accuracy": 90, "citation": 90, "overall": 90, "rationale": "ok"}

        return _Score()

    monkeypatch.setattr(eval_run.bootstrap.app_state, "graph", _FakeGraph())
    monkeypatch.setattr(eval_run, "judge_one", _fake_judge_one)

    record = asyncio.run(eval_run._run_one({"id": "tech_01", "query": "q"}, "run-x"))

    assert record["node_metrics"]["llm_calls"] == 2
    assert record["node_metrics"]["prompt_tokens"] == 6
    assert record["node_metrics"]["by_agent"]["planner"]["llm_calls"] == 2
    assert any(item["id"] == "trace_observability" and item["status"] == "covered" for item in record["framework_coverage"])


def _fake_llm_result(*, prompt_tokens: int, completion_tokens: int):
    class _Generation:
        message = type(
            "Message",
            (),
            {
                "response_metadata": {
                    "token_usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                }
            },
        )()

    class _Result:
        llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
        generations = [[_Generation()]]

    return _Result()
