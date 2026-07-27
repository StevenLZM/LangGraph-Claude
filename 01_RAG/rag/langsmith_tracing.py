"""Safe payload normalization for LangSmith tracing."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from langchain_core.documents import Document


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
