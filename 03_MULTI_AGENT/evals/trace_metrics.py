"""Local callback trace metrics for eval runs.

LangSmith receives the full trace when configured. This collector records a
small local metric summary into results.jsonl so production gates can work even
without querying LangSmith's API.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class EvalTraceCollector(BaseCallbackHandler):
    """Collect chain, LLM, and tool metrics from LangChain callbacks."""

    def __init__(self, *, case_id: str):
        self.case_id = case_id
        self._runs: dict[str, dict[str, Any]] = {}
        self._by_agent: dict[str, dict[str, int | float]] = defaultdict(_metric_bucket)
        self._totals = _metric_bucket()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = _agent_from(tags=tags, metadata=metadata, serialized=serialized)
        agent = self._start(
            str(run_id),
            kind="chain",
            agent=agent,
            parent_run_id=parent_run_id,
            explicit_agent=_has_explicit_agent(tags=tags, metadata=metadata),
        )
        self._inc(agent, "chain_runs")

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        self._finish(str(run_id))

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = self._runs.get(str(run_id), {}).get("agent", "unknown")
        self._inc(agent, "chain_errors")
        self._finish(str(run_id))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = _agent_from(tags=tags, metadata=metadata, serialized=serialized)
        agent = self._start(
            str(run_id),
            kind="llm",
            agent=agent,
            parent_run_id=parent_run_id,
            explicit_agent=_has_explicit_agent(tags=tags, metadata=metadata),
        )
        self._inc(agent, "llm_calls")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        self.on_llm_start(
            serialized,
            [],
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        run = self._runs.get(str(run_id), {})
        agent = run.get("agent", "unknown")
        usage = _extract_token_usage(response)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self._add(agent, key, int(usage.get(key) or 0))
        self._finish(str(run_id))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = self._runs.get(str(run_id), {}).get("agent", "unknown")
        self._inc(agent, "llm_errors")
        self._finish(str(run_id))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = _agent_from(tags=tags, metadata=metadata, serialized=serialized)
        agent = self._start(
            str(run_id),
            kind="tool",
            agent=agent,
            parent_run_id=parent_run_id,
            explicit_agent=_has_explicit_agent(tags=tags, metadata=metadata),
        )
        self._inc(agent, "tool_calls")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        self._finish(str(run_id))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        agent = self._runs.get(str(run_id), {}).get("agent", "unknown")
        self._inc(agent, "tool_errors")
        self._finish(str(run_id))

    def summary(self) -> dict[str, Any]:
        out = {"case_id": self.case_id, **self._totals}
        out["by_agent"] = {
            agent: dict(metrics)
            for agent, metrics in sorted(self._by_agent.items())
            if any(metrics.values())
        }
        return out

    def _start(
        self,
        run_id: str,
        *,
        kind: str,
        agent: str,
        parent_run_id: Any | None = None,
        explicit_agent: bool = False,
    ) -> str:
        if parent_run_id is not None and not explicit_agent:
            parent = self._runs.get(str(parent_run_id))
            if parent and parent.get("agent"):
                agent = str(parent["agent"])
        self._runs[run_id] = {
            "kind": kind,
            "agent": agent,
            "started": time.perf_counter(),
        }
        return agent

    def _finish(self, run_id: str) -> None:
        run = self._runs.pop(run_id, None)
        if not run:
            return
        elapsed_ms = round((time.perf_counter() - run["started"]) * 1000, 1)
        agent = run.get("agent", "unknown")
        kind = run.get("kind", "run")
        self._add(agent, f"{kind}_elapsed_ms", elapsed_ms)
        self._add(agent, "total_elapsed_ms", elapsed_ms)

    def _inc(self, agent: str, key: str) -> None:
        self._add(agent, key, 1)

    def _add(self, agent: str, key: str, value: int | float) -> None:
        self._totals[key] = self._totals.get(key, 0) + value
        self._by_agent[agent][key] = self._by_agent[agent].get(key, 0) + value


def _metric_bucket() -> dict[str, int | float]:
    return {
        "chain_runs": 0,
        "chain_errors": 0,
        "llm_calls": 0,
        "llm_errors": 0,
        "tool_calls": 0,
        "tool_errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "chain_elapsed_ms": 0.0,
        "llm_elapsed_ms": 0.0,
        "tool_elapsed_ms": 0.0,
        "total_elapsed_ms": 0.0,
    }


def _agent_from(
    *,
    tags: list[str] | None,
    metadata: dict[str, Any] | None,
    serialized: dict[str, Any],
) -> str:
    if metadata and metadata.get("agent"):
        return str(metadata["agent"])
    for tag in tags or []:
        if tag.startswith("agent:"):
            return tag.split(":", 1)[1]
    return str(serialized.get("name") or serialized.get("id") or "unknown")


def _has_explicit_agent(*, tags: list[str] | None, metadata: dict[str, Any] | None) -> bool:
    if metadata and metadata.get("agent"):
        return True
    return any(tag.startswith("agent:") for tag in tags or [])


def _extract_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "llm_output", None) or {}
    token_usage = usage.get("token_usage") or usage.get("usage") or {}
    if token_usage:
        return _normalize_usage(token_usage)

    for generations in getattr(response, "generations", []) or []:
        for generation in generations or []:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", None) or {}
            token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
            if token_usage:
                return _normalize_usage(token_usage)
    return {}


def _normalize_usage(raw: dict[str, Any]) -> dict[str, int]:
    prompt = raw.get("prompt_tokens") or raw.get("input_tokens") or 0
    completion = raw.get("completion_tokens") or raw.get("output_tokens") or 0
    total = raw.get("total_tokens") or (prompt + completion)
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(total or 0),
    }
