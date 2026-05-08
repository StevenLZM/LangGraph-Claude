from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ObservabilityMetrics:
    request_count: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    token_usage: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    response_time_count: float = 0.0
    response_time_sum: float = 0.0
    quality_score_count: float = 0.0
    quality_score_sum: float = 0.0
    active_sessions: set[str] = field(default_factory=set)
    error_count: float = 0.0
    transferred_count: float = 0.0


_metrics = ObservabilityMetrics()
_lock = threading.Lock()


def record_chat_request(
    *,
    status: str,
    session_id: str,
    response_time_ms: int,
    token_used: int,
    quality_score: int | None,
    needs_human_transfer: bool = False,
) -> None:
    with _lock:
        _metrics.request_count[status] += 1.0
        _metrics.response_time_count += 1.0
        _metrics.response_time_sum += max(0, response_time_ms) / 1000
        _metrics.token_usage["estimated"] += max(0, token_used)
        if quality_score is not None:
            _metrics.quality_score_count += 1.0
            _metrics.quality_score_sum += max(0, min(100, quality_score))
        if session_id:
            _metrics.active_sessions.add(session_id)
        if status == "error":
            _metrics.error_count += 1.0
        if needs_human_transfer:
            _metrics.transferred_count += 1.0


def record_chat_error(*, status: str, session_id: str = "") -> None:
    with _lock:
        _metrics.request_count[status] += 1.0
        if status == "error":
            _metrics.error_count += 1.0
        if session_id:
            _metrics.active_sessions.add(session_id)


def reset_observability_metrics() -> None:
    with _lock:
        global _metrics
        _metrics = ObservabilityMetrics()


def collect_health_metrics() -> dict[str, str]:
    return {"metrics": "configured"}


def render_prometheus_metrics() -> str:
    with _lock:
        lines = [
            "# HELP agent_requests_total Total chat requests.",
            "# TYPE agent_requests_total counter",
        ]
        for status in sorted(_metrics.request_count):
            lines.append(
                f'agent_requests_total{{status="{_escape_label(status)}"}} '
                f"{_format_number(_metrics.request_count[status])}"
            )

        lines.extend(
            [
                "# HELP agent_response_time_seconds Chat response time in seconds.",
                "# TYPE agent_response_time_seconds summary",
                f"agent_response_time_seconds_count {_format_number(_metrics.response_time_count)}",
                f"agent_response_time_seconds_sum {_format_number(_metrics.response_time_sum)}",
                "# HELP agent_tokens_total Estimated token usage.",
                "# TYPE agent_tokens_total counter",
            ]
        )
        for token_type in sorted(_metrics.token_usage):
            lines.append(
                f'agent_tokens_total{{type="{_escape_label(token_type)}"}} '
                f"{_format_number(_metrics.token_usage[token_type])}"
            )

        lines.extend(
            [
                "# HELP agent_active_sessions Current active sessions.",
                "# TYPE agent_active_sessions gauge",
                f"agent_active_sessions {_format_number(float(len(_metrics.active_sessions)))}",
                "# HELP agent_quality_score Answer quality score.",
                "# TYPE agent_quality_score summary",
                f"agent_quality_score_count {_format_number(_metrics.quality_score_count)}",
                f"agent_quality_score_sum {_format_number(_metrics.quality_score_sum)}",
                "# HELP agent_errors_total Total error requests.",
                "# TYPE agent_errors_total counter",
                f"agent_errors_total {_format_number(_metrics.error_count)}",
                "# HELP agent_human_transfers_total Total human transfer requests.",
                "# TYPE agent_human_transfers_total counter",
                f"agent_human_transfers_total {_format_number(_metrics.transferred_count)}",
            ]
        )
        return "\n".join(lines) + "\n"


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_number(value: float) -> str:
    if value == int(value):
        return f"{value:.1f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")
