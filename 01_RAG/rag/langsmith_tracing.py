"""Credential-safe, failure-isolated LangSmith tracing helpers."""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langsmith import Client, trace, tracing_context


REDACTED_CREDENTIAL = "[REDACTED_CREDENTIAL]"
_CIRCULAR_REFERENCE = "[CIRCULAR_REFERENCE]"
_MAX_UNKNOWN_REPR_LENGTH = 256
_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "password",
        "passwd",
        "token",
        "accesstoken",
        "refreshtoken",
        "secret",
        "clientsecret",
        "cookie",
    }
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangSmithSettings:
    """LangSmith configuration resolved without exposing secrets in logs."""

    enabled: bool
    api_key: str
    project_name: str


def resolve_langsmith_settings(
    environ: Mapping[str, str] | None = None,
) -> LangSmithSettings:
    """Resolve LangSmith settings, preferring current names over compatibility ones."""
    environment = os.environ if environ is None else environ
    trace_flag = _prefer_environment_value(
        environment, "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"
    )
    api_key = _prefer_environment_value(
        environment, "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"
    )
    project_name = _prefer_environment_value(
        environment, "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"
    )
    return LangSmithSettings(
        enabled=_is_enabled_flag(trace_flag) and bool(api_key),
        api_key=api_key,
        project_name=project_name,
    )


def _prefer_environment_value(
    environment: Mapping[str, str], primary: str, compatibility: str
) -> str:
    if primary in environment:
        return environment[primary]
    return environment.get(compatibility, "")


def _is_enabled_flag(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _build_safe_langsmith_client(settings: LangSmithSettings) -> Client:
    """Construct a configured client once for each process-local setting set."""
    return Client(
        api_key=settings.api_key,
        hide_inputs=sanitize_trace_credentials,
        hide_outputs=sanitize_trace_credentials,
        hide_metadata=sanitize_trace_credentials,
    )


def get_safe_langsmith_client(
    settings: LangSmithSettings | None = None,
) -> Client | None:
    """Return a cached safe client, or no client when tracing is unavailable."""
    resolved_settings = settings or resolve_langsmith_settings()
    if not resolved_settings.enabled or not resolved_settings.api_key:
        return None
    try:
        return _build_safe_langsmith_client(resolved_settings)
    except Exception:
        _LOGGER.warning("LangSmith client setup failed; tracing is disabled", exc_info=True)
        return None


@contextmanager
def rag_tracing_context(
    *,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Scope LangSmith request tracing without affecting business execution."""
    settings = resolve_langsmith_settings()
    client = get_safe_langsmith_client(settings)
    if client is None:
        yield None
        return

    safe_metadata = sanitize_trace_credentials(metadata) if metadata is not None else None
    with _failure_isolated_context(
        lambda: tracing_context(
            client=client,
            project_name=settings.project_name,
            enabled=True,
            tags=list(tags),
            metadata=safe_metadata,
        )
    ):
        yield None


@contextmanager
def trace_span(
    name: str,
    *,
    run_type: str = "chain",
    inputs: Mapping[str, Any] | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Yield a LangSmith span when available, otherwise yield ``None``."""
    settings = resolve_langsmith_settings()
    client = get_safe_langsmith_client(settings)
    if client is None:
        yield None
        return

    safe_inputs = sanitize_trace_credentials(inputs) if inputs is not None else None
    safe_metadata = sanitize_trace_credentials(metadata) if metadata is not None else None
    with _failure_isolated_context(
        lambda: trace(
            name=name,
            run_type=run_type,
            client=client,
            inputs=safe_inputs,
            tags=list(tags),
            metadata=safe_metadata,
        )
    ) as span:
        yield span


def end_trace_span(span: Any | None, *, outputs: Mapping[str, Any]) -> None:
    """End a trace span without letting telemetry transport errors escape."""
    if span is None:
        return
    try:
        span.end(outputs=sanitize_trace_credentials(outputs))
    except Exception:
        _LOGGER.warning("LangSmith span end failed", exc_info=True)


@contextmanager
def _failure_isolated_context(factory: Callable[[], Any]) -> Iterator[Any | None]:
    """Convert tracing-context failures into no-ops while preserving business errors."""
    try:
        context = factory()
        span = context.__enter__()
    except Exception:
        _LOGGER.warning("LangSmith tracing setup failed", exc_info=True)
        yield None
        return

    try:
        yield span
    except BaseException:
        exception_info = sys.exc_info()
        try:
            context.__exit__(*exception_info)
        except Exception:
            _LOGGER.warning("LangSmith tracing teardown failed", exc_info=True)
        raise
    else:
        try:
            context.__exit__(None, None, None)
        except Exception:
            _LOGGER.warning("LangSmith tracing teardown failed", exc_info=True)


def _normalize_key(key: object) -> str:
    return re.sub(r"[_-]", "", str(key)).casefold()


def _is_credential_key(key: object) -> bool:
    normalized = _normalize_key(key)
    return normalized in _CREDENTIAL_KEYS or any(
        normalized.endswith(marker) for marker in _CREDENTIAL_KEYS
    )


def sanitize_trace_credentials(payload: Any) -> Any:
    """Return a copy of *payload* with only credential values redacted."""
    return _sanitize_trace_value(payload, seen=set())


def serialize_documents_for_trace(
    documents: Sequence[Document],
) -> list[dict[str, Any]]:
    """Preserve complete document content and metadata in an audit payload."""
    return [
        {
            "page_content": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in documents
    ]


def _sanitize_trace_value(value: Any, *, seen: set[int]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    value_id = id(value)
    if value_id in seen:
        return _CIRCULAR_REFERENCE

    if isinstance(value, Document):
        return _sanitize_container(
            value,
            seen=seen,
            build=lambda: {
                "page_content": _sanitize_trace_value(value.page_content, seen=seen),
                "metadata": _sanitize_trace_value(value.metadata, seen=seen),
            },
        )

    if isinstance(value, Mapping):
        return _sanitize_container(
            value,
            seen=seen,
            build=lambda: {
                key: (
                    REDACTED_CREDENTIAL
                    if _is_credential_key(key)
                    else _sanitize_trace_value(item, seen=seen)
                )
                for key, item in value.items()
            },
        )

    if isinstance(value, list):
        return _sanitize_container(
            value,
            seen=seen,
            build=lambda: [_sanitize_trace_value(item, seen=seen) for item in value],
        )

    if isinstance(value, tuple):
        return _sanitize_container(
            value,
            seen=seen,
            build=lambda: tuple(_sanitize_trace_value(item, seen=seen) for item in value),
        )

    if is_dataclass(value) and not isinstance(value, type):
        return _sanitize_container(
            value,
            seen=seen,
            build=lambda: {
                field.name: (
                    REDACTED_CREDENTIAL
                    if _is_credential_key(field.name)
                    else _sanitize_trace_value(getattr(value, field.name), seen=seen)
                )
                for field in fields(value)
                if not field.name.startswith("_")
            },
        )

    return _bounded_repr(value)


def _sanitize_container(
    value: Any, *, seen: set[int], build: Callable[[], Any]
) -> Any:
    seen.add(id(value))
    try:
        return build()
    finally:
        seen.remove(id(value))


def _bounded_repr(value: Any) -> str:
    try:
        representation = repr(value)
    except Exception:
        representation = f"<{type(value).__name__}>"
    return representation[:_MAX_UNKNOWN_REPR_LENGTH]
