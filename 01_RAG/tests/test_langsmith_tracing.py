from __future__ import annotations

from collections import UserDict
from copy import deepcopy
from dataclasses import dataclass

import pytest
from langchain_core.documents import Document

from rag.langsmith_tracing import REDACTED_CREDENTIAL, sanitize_trace_credentials


@pytest.mark.parametrize(
    "credential_key",
    [
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "client_secret",
        "cookie",
        "DASHSCOPE_API_KEY",
        "ES-PASSWORD",
        "custom_access_token",
        "Client-Secret",
    ],
)
def test_sanitize_trace_credentials_redacts_credential_key_variants(credential_key):
    payload = {credential_key: "unique-credential-value"}

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized == {credential_key: REDACTED_CREDENTIAL}
    assert payload == {credential_key: "unique-credential-value"}


def test_sanitize_trace_credentials_redacts_nested_credential_fields():
    payload = {
        "question": "保修期多久？",
        "DASHSCOPE_API_KEY": "unique-api-key-value",
        "headers": {
            "Authorization": "Bearer unique-token",
            "client-secret": "unique-client-secret",
        },
        "items": [
            {"refresh_token": "unique-refresh-token"},
            {"cookie": "unique-cookie"},
        ],
    }

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized["DASHSCOPE_API_KEY"] == REDACTED_CREDENTIAL
    assert sanitized["headers"]["Authorization"] == REDACTED_CREDENTIAL
    assert sanitized["headers"]["client-secret"] == REDACTED_CREDENTIAL
    assert sanitized["items"][0]["refresh_token"] == REDACTED_CREDENTIAL
    assert sanitized["items"][1]["cookie"] == REDACTED_CREDENTIAL


def test_sanitizer_preserves_audit_content_without_mutating_input():
    payload = {
        "question": "unique_question_text",
        "history": [{"role": "user", "content": "历史问题"}],
        "page_content": "unique_chunk_secret_text",
        "context": "完整证据 [S1]",
        "answer": "保修期十二个月 [S1]",
        "metadata": {
            "doc_id": "doc-1",
            "parent_id": "parent-1",
            "rerank_score": 0.91,
        },
        "scores": {"dense": 0.8, "rerank": 0.91},
    }
    original = deepcopy(payload)

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized == original
    assert payload == original
    assert sanitized is not payload
    assert sanitized["history"] is not payload["history"]
    assert sanitized["metadata"] is not payload["metadata"]


@dataclass
class _TraceRecord:
    answer: str
    token: str
    metadata: dict[str, str]


class _StatefulUnknown:
    def __init__(self) -> None:
        self.repr_calls = 0

    def __repr__(self) -> str:
        self.repr_calls += 1
        return "unknown-object-" + ("x" * 400)


def test_sanitizer_serializes_supported_containers_documents_and_dataclasses():
    document = Document(
        page_content="保修期为 12 个月",
        metadata={"doc_id": "doc-1", "api_key": "document-api-key"},
    )
    record = _TraceRecord(
        answer="保修期十二个月 [S1]",
        token="record-token",
        metadata={"score": "0.91"},
    )
    payload = UserDict(
        {
            "tuple": ("first", {"password": "tuple-password"}),
            "list": [None, 3, True],
            "document": document,
            "record": record,
        }
    )

    sanitized = sanitize_trace_credentials(payload)

    assert type(sanitized) is dict
    assert isinstance(sanitized["tuple"], tuple)
    assert sanitized["tuple"][1]["password"] == REDACTED_CREDENTIAL
    assert isinstance(sanitized["list"], list)
    assert sanitized["list"] == [None, 3, True]
    assert sanitized["document"] == {
        "page_content": "保修期为 12 个月",
        "metadata": {"doc_id": "doc-1", "api_key": REDACTED_CREDENTIAL},
    }
    assert sanitized["record"] == {
        "answer": "保修期十二个月 [S1]",
        "token": REDACTED_CREDENTIAL,
        "metadata": {"score": "0.91"},
    }
    assert document.metadata["api_key"] == "document-api-key"
    assert record.token == "record-token"


def test_sanitizer_uses_bounded_repr_for_unknown_objects_without_business_methods():
    unknown = _StatefulUnknown()

    sanitized = sanitize_trace_credentials({"unknown": unknown})

    assert isinstance(sanitized["unknown"], str)
    assert sanitized["unknown"].startswith("unknown-object-")
    assert len(sanitized["unknown"]) <= 256
    assert unknown.repr_calls == 1


def test_sanitizer_terminates_for_self_referential_containers():
    payload: dict[str, object] = {"question": "循环输入"}
    payload["self"] = payload

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized["question"] == "循环输入"
    assert isinstance(sanitized["self"], str)
    assert sanitized["self"] == "[CIRCULAR_REFERENCE]"
