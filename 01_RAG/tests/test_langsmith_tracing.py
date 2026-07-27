from __future__ import annotations

from collections import UserDict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from threading import Barrier

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import get_tracing_context
from pydantic import BaseModel

from rag import langsmith_tracing as tracing
from rag.chain import RagExecution, create_rag_chain
from rag.langsmith_tracing import REDACTED_CREDENTIAL, sanitize_trace_credentials
from rag.retrieval_trace import EvaluationTrace
from rag.retriever import RetrievalPipelineComponents, retrieve_with_trace


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


class _TracePayloadModel(BaseModel):
    question: str
    auth_context: dict[str, object]
    metadata: dict[str, object]


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


def test_sanitizer_preserves_complete_messages_and_redacts_structured_fields():
    human_content = "用户历史问题：" + ("保修条款甲乙丙" * 1_000)
    ai_content = "助手历史回答：" + ("完整审计内容丁戊己" * 1_000)
    human = HumanMessage(
        content=human_content,
        additional_kwargs={
            "authorization": "Bearer message-authorization",
            "tool_context": {"client_secret": "message-client-secret"},
        },
        response_metadata={
            "model": "offline-model",
            "refresh_token": "message-refresh-token",
        },
        name="customer",
        id="human-message-1",
    )
    ai = AIMessage(
        content=ai_content,
        additional_kwargs={
            "tool_calls": [
                {
                    "name": "lookup",
                    "args": {"api_key": "message-tool-api-key"},
                }
            ]
        },
        response_metadata={
            "usage": {"input_tokens": 12},
            "cookie": "message-cookie",
        },
        name="assistant",
        id="ai-message-1",
    )
    original_human = human.model_dump(mode="python")
    original_ai = ai.model_dump(mode="python")

    sanitized = sanitize_trace_credentials(
        {"chat_history": [human, ai]}
    )["chat_history"]

    assert sanitized[0]["type"] == "human"
    assert sanitized[0]["content"].encode("utf-8") == human_content.encode("utf-8")
    assert sanitized[0]["name"] == "customer"
    assert sanitized[0]["id"] == "human-message-1"
    assert (
        sanitized[0]["additional_kwargs"]["authorization"]
        == REDACTED_CREDENTIAL
    )
    assert (
        sanitized[0]["additional_kwargs"]["tool_context"]["client_secret"]
        == REDACTED_CREDENTIAL
    )
    assert (
        sanitized[0]["response_metadata"]["refresh_token"]
        == REDACTED_CREDENTIAL
    )
    assert sanitized[1]["type"] == "ai"
    assert sanitized[1]["content"].encode("utf-8") == ai_content.encode("utf-8")
    assert sanitized[1]["name"] == "assistant"
    assert sanitized[1]["id"] == "ai-message-1"
    assert (
        sanitized[1]["additional_kwargs"]["tool_calls"][0]["args"]["api_key"]
        == REDACTED_CREDENTIAL
    )
    assert sanitized[1]["response_metadata"]["cookie"] == REDACTED_CREDENTIAL
    assert human.model_dump(mode="python") == original_human
    assert ai.model_dump(mode="python") == original_ai


def test_sanitizer_uses_public_pydantic_dump_without_mutating_model():
    model = _TracePayloadModel(
        question="保修期多久？",
        auth_context={
            "tenant_id": "tenant-1",
            "access_token": "model-access-token",
        },
        metadata={
            "nested": {
                "password": "model-password",
                "score": 0.91,
            }
        },
    )
    original = model.model_dump(mode="python")

    sanitized = sanitize_trace_credentials(model)

    assert sanitized == {
        "question": "保修期多久？",
        "auth_context": {
            "tenant_id": "tenant-1",
            "access_token": REDACTED_CREDENTIAL,
        },
        "metadata": {
            "nested": {
                "password": REDACTED_CREDENTIAL,
                "score": 0.91,
            }
        },
    }
    assert model.model_dump(mode="python") == original


def test_sanitizer_uses_type_only_marker_without_invoking_unknown_repr():
    unknown = _StatefulUnknown()

    sanitized = sanitize_trace_credentials({"unknown": unknown})

    assert sanitized["unknown"] == "[UNSUPPORTED_TYPE:_StatefulUnknown]"
    assert unknown.repr_calls == 0


def test_sanitizer_terminates_for_self_referential_containers():
    payload: dict[str, object] = {"question": "循环输入"}
    payload["self"] = payload

    sanitized = sanitize_trace_credentials(payload)

    assert sanitized["question"] == "循环输入"
    assert isinstance(sanitized["self"], str)
    assert sanitized["self"] == "[CIRCULAR_REFERENCE]"


def test_tracing_is_disabled_without_flag_or_api_key():
    """Removing either opt-in input must prevent tracing from starting."""
    assert tracing.resolve_langsmith_settings({}).enabled is False
    assert tracing.resolve_langsmith_settings({"LANGSMITH_TRACING": "true"}).enabled is False
    assert tracing.resolve_langsmith_settings({"LANGSMITH_API_KEY": "test-key"}).enabled is False


def test_settings_accept_langchain_compatibility_names():
    """Dropping the legacy LANGCHAIN names would silently disable existing users."""
    settings = tracing.resolve_langsmith_settings(
        {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_API_KEY": "test-key",
            "LANGCHAIN_PROJECT": "legacy-project",
        }
    )

    assert settings.enabled is True
    assert settings.api_key == "test-key"
    assert settings.project_name == "legacy-project"


def test_langsmith_environment_names_take_precedence_over_langchain_compatibility_names():
    """A migrated deployment must not be controlled by stale compatibility values."""
    settings = tracing.resolve_langsmith_settings(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "current-key",
            "LANGSMITH_PROJECT": "current-project",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGCHAIN_API_KEY": "legacy-key",
            "LANGCHAIN_PROJECT": "legacy-project",
        }
    )

    assert settings == tracing.LangSmithSettings(True, "current-key", "current-project")


class _FakeClient:
    instances: list["_FakeClient"] = []
    construction_error: Exception | None = None

    def __init__(self, **kwargs):
        if self.construction_error is not None:
            raise self.construction_error
        self.kwargs = kwargs
        self.instances.append(self)


@pytest.fixture
def fake_client(monkeypatch):
    """Replace only the networked LangSmith client boundary."""
    _FakeClient.instances = []
    _FakeClient.construction_error = None
    tracing._build_safe_langsmith_client.cache_clear()
    monkeypatch.setattr(tracing, "Client", _FakeClient)
    yield _FakeClient
    tracing._build_safe_langsmith_client.cache_clear()


def test_safe_client_applies_sanitizer_to_all_payload_channels(fake_client):
    """Removing a hide callback would leak automatically traced payloads."""
    settings = tracing.LangSmithSettings(True, "test-key", "project")

    client = tracing.get_safe_langsmith_client(settings)

    assert client.kwargs["hide_inputs"] is sanitize_trace_credentials
    assert client.kwargs["hide_outputs"] is sanitize_trace_credentials
    assert client.kwargs["hide_metadata"] is sanitize_trace_credentials


def test_disabled_settings_do_not_construct_a_langsmith_client(fake_client):
    """Creating a client while disabled can trigger unwanted network setup."""
    client = tracing.get_safe_langsmith_client(
        tracing.LangSmithSettings(False, "test-key", "project")
    )

    assert client is None
    assert fake_client.instances == []


def test_safe_client_is_reused_for_the_same_settings(fake_client):
    """Removing client caching would duplicate process-local client resources."""
    settings = tracing.LangSmithSettings(True, "test-key", "project")

    first = tracing.get_safe_langsmith_client(settings)
    second = tracing.get_safe_langsmith_client(settings)

    assert first is second
    assert fake_client.instances == [first]


def test_client_construction_failure_is_converted_to_no_op(fake_client):
    """A failed telemetry client construction must not fail request handling."""
    fake_client.construction_error = RuntimeError("transport unavailable")

    client = tracing.get_safe_langsmith_client(
        tracing.LangSmithSettings(True, "test-key", "project")
    )

    assert client is None


class _RecordingContext:
    def __init__(self, *, span=None, enter_error=None, exit_error=None):
        self.span = span
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.exit_calls = []

    def __enter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self.span

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_calls.append((exc_type, exc_value, traceback))
        if self.exit_error is not None:
            raise self.exit_error
        return False


def _enabled_settings():
    return tracing.LangSmithSettings(True, "test-key", "project")


def test_trace_span_is_noop_when_disabled(monkeypatch):
    """Disabled tracing must avoid both client construction and trace setup."""
    monkeypatch.setattr(
        tracing,
        "resolve_langsmith_settings",
        lambda environ=None: tracing.LangSmithSettings(False, "", ""),
    )
    called = []

    with tracing.trace_span("rag.request") as span:
        called.append(span)
    with tracing.rag_tracing_context():
        called.append(None)
    tracing.end_trace_span(None, outputs={"token": "test-token"})

    assert called == [None, None]


def test_rag_context_keeps_ambient_disabled_when_safe_context_entry_fails(
    monkeypatch,
):
    """A failed enabled context must not restore ambient tracing mid-request."""
    from langsmith import tracing_context as real_tracing_context

    monkeypatch.setattr(
        tracing,
        "resolve_langsmith_settings",
        lambda environ=None: _enabled_settings(),
    )
    monkeypatch.setattr(
        tracing,
        "get_safe_langsmith_client",
        lambda settings: object(),
    )

    def fail_enabled_context(**kwargs):
        if kwargs["enabled"] is True:
            raise RuntimeError("safe context unavailable")
        return real_tracing_context(**kwargs)

    monkeypatch.setattr(tracing, "tracing_context", fail_enabled_context)

    with real_tracing_context(enabled=True):
        with tracing.rag_tracing_context():
            assert get_tracing_context()["enabled"] is False


def test_rag_context_sanitizes_explicit_metadata_before_transport(monkeypatch):
    """Passing raw context metadata would bypass the automatic client hooks."""
    received = []
    context = _RecordingContext()
    client = object()
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: client)
    monkeypatch.setattr(
        tracing,
        "tracing_context",
        lambda **kwargs: received.append(kwargs) or context,
    )

    with tracing.rag_tracing_context(metadata={"api_key": "test-key", "answer": "safe"}):
        pass

    assert received == [
        {"enabled": False},
        {
            "client": client,
            "project_name": "project",
            "enabled": True,
            "tags": [],
            "metadata": {"api_key": REDACTED_CREDENTIAL, "answer": "safe"},
        }
    ]


def test_rag_context_converts_tuple_tags_to_a_langsmith_compatible_list(monkeypatch):
    """Tuple tags must work when LangSmith appends its own list of tags."""
    received = []
    context = _RecordingContext()
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())

    def langsmith_context(**kwargs):
        received.append(kwargs)
        if kwargs["enabled"] is True:
            kwargs["tags"] + ["langsmith"]
        return context

    monkeypatch.setattr(tracing, "tracing_context", langsmith_context)

    with tracing.rag_tracing_context(tags=("request",)):
        pass

    assert received[1]["tags"] == ["request"]
    assert context.exit_calls == [(None, None, None), (None, None, None)]


def test_span_sanitizes_explicit_inputs_metadata_and_outputs(monkeypatch):
    """Removing explicit sanitization would leak data sent through manual spans."""
    received = []

    class Span:
        def __init__(self):
            self.ended_with = None

        def end(self, *, outputs):
            self.ended_with = outputs

    span = Span()
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())
    monkeypatch.setattr(
        tracing,
        "trace",
        lambda **kwargs: received.append(kwargs) or _RecordingContext(span=span),
    )

    with tracing.trace_span(
        "rag.request",
        inputs={"token": "test-token", "question": "question"},
        metadata={"password": "test-password"},
    ) as actual_span:
        tracing.end_trace_span(actual_span, outputs={"cookie": "test-cookie", "answer": "answer"})

    assert received[0]["inputs"] == {"token": REDACTED_CREDENTIAL, "question": "question"}
    assert received[0]["metadata"] == {"password": REDACTED_CREDENTIAL}
    assert span.ended_with == {"cookie": REDACTED_CREDENTIAL, "answer": "answer"}


def test_trace_span_converts_tuple_tags_to_a_langsmith_compatible_list(monkeypatch):
    """Tuple span tags must not degrade tracing when LangSmith appends a list."""
    received = []
    span = object()
    context = _RecordingContext(span=span)
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())

    def langsmith_trace(**kwargs):
        received.append(kwargs)
        kwargs["tags"] + ["langsmith"]
        return context

    monkeypatch.setattr(tracing, "trace", langsmith_trace)

    with tracing.trace_span("rag.request", tags=("rag",)) as actual_span:
        assert actual_span is span

    assert received[0]["tags"] == ["rag"]
    assert context.exit_calls == [(None, None, None)]


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_rag_context_transport_failures_preserve_the_business_result(monkeypatch, phase):
    """An error while opening or closing context must not replace a successful result."""
    disabled_context = _RecordingContext()
    enabled_context = _RecordingContext(
        **{f"{phase}_error": RuntimeError("transport unavailable")}
    )
    calls = 0
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())
    monkeypatch.setattr(
        tracing,
        "tracing_context",
        lambda **kwargs: (
            enabled_context if kwargs["enabled"] is True else disabled_context
        ),
    )

    def business_function():
        nonlocal calls
        calls += 1
        with tracing.rag_tracing_context():
            return "business-result"

    assert business_function() == "business-result"
    assert calls == 1


@pytest.mark.parametrize("phase", ["enter", "exit"])
def test_trace_span_transport_failures_preserve_the_business_result(monkeypatch, phase):
    """A trace context-manager error must not replace a successful result."""
    context = _RecordingContext(**{f"{phase}_error": RuntimeError("transport unavailable")})
    calls = 0
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())
    monkeypatch.setattr(tracing, "trace", lambda **kwargs: context)

    def business_function():
        nonlocal calls
        calls += 1
        with tracing.trace_span("rag.request"):
            return "business-result"

    assert business_function() == "business-result"
    assert calls == 1


def test_span_end_failure_preserves_the_business_result(monkeypatch):
    """A failed end call must not replace the enclosing business result."""
    calls = 0

    class FailingSpan:
        def end(self, *, outputs):
            raise RuntimeError("transport unavailable")

    def business_function():
        nonlocal calls
        calls += 1
        tracing.end_trace_span(FailingSpan(), outputs={"answer": "business-result"})
        return "business-result"

    assert business_function() == "business-result"
    assert calls == 1


def test_trace_span_preserves_a_business_exception_when_trace_exit_also_fails(monkeypatch):
    """Suppressing the business exception would hide the actual request failure."""
    context = _RecordingContext(exit_error=RuntimeError("transport unavailable"))
    monkeypatch.setattr(tracing, "resolve_langsmith_settings", lambda environ=None: _enabled_settings())
    monkeypatch.setattr(tracing, "get_safe_langsmith_client", lambda settings: object())
    monkeypatch.setattr(tracing, "trace", lambda **kwargs: context)

    class BusinessError(Exception):
        pass

    with pytest.raises(BusinessError, match="business failure"):
        with tracing.trace_span("rag.request"):
            raise BusinessError("business failure")


def test_parallel_recall_copies_trace_context_to_each_worker():
    """Losing the request context would detach both recall spans from retrieval."""
    marker = ContextVar("retrieval_trace_marker", default=None)
    workers_ready = Barrier(2)
    seen_by_bm25 = []
    seen_by_dense = []

    def child(child_id):
        return Document(
            page_content=f"child {child_id}",
            metadata={
                "doc_id": "doc-1",
                "parent_id": f"parent-{child_id}",
                "child_id": child_id,
                "source": "manual.pdf",
                "page_range": "3",
            },
        )

    def parent(parent_id):
        return Document(
            page_content=f"parent {parent_id}",
            metadata={
                "doc_id": "doc-1",
                "parent_id": parent_id,
                "source": "manual.pdf",
                "page_range": "3",
            },
        )

    def bm25(query, context):
        workers_ready.wait(timeout=5)
        seen_by_bm25.append(marker.get())
        return [child("bm25")]

    def dense(query, context):
        workers_ready.wait(timeout=5)
        seen_by_dense.append(marker.get())
        return [child("dense")]

    components = RetrievalPipelineComponents(
        bm25=bm25,
        dense=dense,
        rrf=lambda bm25_documents, dense_documents: [child("rrf")],
        cross_encoder=lambda query, documents: [child("reranked")],
        business_fusion=lambda query, documents, context: [child("business")],
        diversify_parents=lambda documents: [parent("diversified")],
        assemble_context=lambda documents: [parent("final")],
    )
    token = marker.set("rag-request-1")
    try:
        retrieve_with_trace(
            "保修期",
            retrieval_context={"trace_id": "trace-1"},
            components=components,
        )
    finally:
        marker.reset(token)

    assert seen_by_bm25 == ["rag-request-1"]
    assert seen_by_dense == ["rag-request-1"]


def test_runnable_forwards_only_standard_config_audit_fields(monkeypatch):
    """Dropping config tags or leaking configurable objects breaks trace isolation."""
    import rag.chain as chain

    received = {}
    execution = RagExecution(
        answer="保修期为 12 个月。[S1]",
        source_documents=(),
        trace=EvaluationTrace(
            trace_id="trace-1",
            original_query="保修期多久？",
            rewritten_query="产品保修期多久？",
        ),
        generation_latency_ms=1.0,
    )

    def fake_run(question, **kwargs):
        received["question"] = question
        received.update(kwargs)
        return execution

    monkeypatch.setattr(chain, "run_rag_with_trace", fake_run)

    result = create_rag_chain().invoke(
        {
            "question": "保修期多久？",
            "auth_context": {"tenant_id": "tenant-1"},
        },
        config={
            "tags": ["01-rag", "interactive"],
            "metadata": {"session_id": "session-1"},
            "configurable": {"session_factory": object()},
        },
    )

    assert result["answer"] == execution.answer
    assert received == {
        "question": "保修期多久？",
        "chat_history": (),
        "auth_context": {"tenant_id": "tenant-1"},
        "trace_tags": ("01-rag", "interactive"),
        "trace_metadata": {"session_id": "session-1"},
    }
