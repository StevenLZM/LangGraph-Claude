from __future__ import annotations

import os
from typing import Any


def configure_langsmith(
    *,
    enabled: bool,
    endpoint: str,
    api_key: str,
    project: str,
) -> None:
    if not enabled:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_PROJECT"] = project
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key


def build_trace_config(
    *,
    session_id: str,
    user_id: str,
    environment: str,
    app_version: str,
) -> dict[str, Any]:
    return {
        "tags": ["customer-service", f"session:{session_id}", f"user:{user_id}"],
        "metadata": {
            "session_id": session_id,
            "user_id": user_id,
            "environment": environment,
            "version": app_version,
        },
    }
