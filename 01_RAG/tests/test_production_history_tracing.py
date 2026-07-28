from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langsmith import get_tracing_context

from rag.langsmith_tracing import REDACTED_CREDENTIAL
from rag.retrieval_trace import EvaluationTrace
from rag.retriever import RetrievalExecution


_CREDENTIAL_MARKER = "synthetic-production-authorization-marker"


class _TracingStateObserver(BaseCallbackHandler):
    def __init__(self) -> None:
        self.chain_start_states: list[bool | str | None] = []
        self.chain_start_events: list[dict[str, Any]] = []

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self.chain_start_states.append(get_tracing_context()["enabled"])
        self.chain_start_events.append(
            {
                "name": kwargs.get("name"),
                "tags": list(kwargs.get("tags") or []),
                "metadata": dict(kwargs.get("metadata") or {}),
            }
        )


class _TransportRecordingClient:
    instances: list["_TransportRecordingClient"] = []
    construction_error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        if self.construction_error is not None:
            raise self.construction_error
        self.hide_inputs = kwargs["hide_inputs"]
        self.hide_outputs = kwargs["hide_outputs"]
        self.hide_metadata = kwargs["hide_metadata"]
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.instances.append(self)

    def create_run(self, **payload: Any) -> None:
        self.created.append(self._transport_payload(payload))

    def update_run(self, run_id: Any, **payload: Any) -> None:
        self.updated.append(
            self._transport_payload({"id": run_id, **payload})
        )

    def _transport_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        transported = dict(payload)
        if transported.get("inputs") is not None:
            transported["inputs"] = self.hide_inputs(transported["inputs"])
        if transported.get("outputs") is not None:
            transported["outputs"] = self.hide_outputs(transported["outputs"])
        extra = transported.get("extra")
        if isinstance(extra, dict):
            transported["extra"] = dict(extra)
            metadata = extra.get("metadata")
            if metadata is not None:
                transported["extra"]["metadata"] = self.hide_metadata(metadata)
        return transported


def _retrieval_execution() -> RetrievalExecution:
    document = Document(
        page_content="产品保修期为 12 个月。",
        metadata={
            "doc_id": "doc-1",
            "parent_id": "parent-1",
            "source": "manual.pdf",
            "page_range": "3",
            "evidence_id": "S1",
        },
    )
    return RetrievalExecution(
        final_documents=(document,),
        trace=EvaluationTrace(
            trace_id="trace-1",
            original_query="保修期多久？",
            rewritten_query="产品保修期多久？",
            metadata={"metadata_filter": {"tenant_id": "tenant-1"}},
        ),
    )


def _configure_ambient_tracing(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ambient-api-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "ambient-project")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)


def _patch_offline_rag_dependencies(monkeypatch, chain) -> None:
    monkeypatch.setattr(
        chain,
        "rewrite_query",
        lambda **kwargs: {
            "rewritten_query": "产品保修期多久？",
            "time_intent": {"type": "none"},
        },
    )
    monkeypatch.setattr(
        chain,
        "retrieve_with_trace",
        lambda *args, **kwargs: _retrieval_execution(),
    )
    monkeypatch.setattr(
        chain,
        "_get_llm",
        lambda model_name=None: RunnableLambda(
            lambda prompt: AIMessage(content="保修期为 12 个月。[S1]")
        ),
    )


def _configure_client_boundary(monkeypatch, tracing, *, fail: bool) -> list[str]:
    import langchain_core.tracers.langchain as langchain_tracer

    default_client_calls: list[str] = []
    _TransportRecordingClient.instances = []
    _TransportRecordingClient.construction_error = (
        RuntimeError("safe client unavailable") if fail else None
    )
    tracing._build_safe_langsmith_client.cache_clear()
    monkeypatch.setattr(tracing, "Client", _TransportRecordingClient)

    def record_default_client():
        default_client_calls.append("default-client")
        raise AssertionError("ambient tracing created a default LangSmith client")

    monkeypatch.setattr(langchain_tracer, "get_client", record_default_client)
    return default_client_calls


def _contains_marker(value: Any) -> bool:
    if isinstance(value, str):
        return _CREDENTIAL_MARKER in value
    if isinstance(value, dict):
        return any(
            _contains_marker(key) or _contains_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_marker(item) for item in value)
    return False


def test_production_history_chain_disables_ambient_before_safe_root_trace(
    monkeypatch,
):
    """Removing the outer boundary would create a default-client parent run."""
    import rag.chain as chain
    from rag import langsmith_tracing as tracing

    _configure_ambient_tracing(monkeypatch)
    _patch_offline_rag_dependencies(monkeypatch, chain)
    default_client_calls = _configure_client_boundary(
        monkeypatch, tracing, fail=False
    )
    observer = _TracingStateObserver()

    history_chain, get_history = chain.create_chain_with_history()
    result = history_chain.invoke(
        {
            "question": "保修期多久？",
            "auth_context": {
                "tenant_id": "tenant-1",
                "authorization": f"Bearer {_CREDENTIAL_MARKER}",
            },
        },
        config={
            "callbacks": [observer],
            "tags": ["01-rag", "interactive"],
            "metadata": {"session_id": "session-1", "channel": "streamlit"},
            "configurable": {"session_id": "session-1"},
        },
    )

    assert result["answer"] == "保修期为 12 个月。[S1]"
    assert observer.chain_start_states[0] is False
    assert default_client_calls == []
    assert len(_TransportRecordingClient.instances) == 1

    client = _TransportRecordingClient.instances[0]
    root_runs = [
        run
        for run in client.created
        if run["name"] == "rag.request" and run.get("parent_run_id") is None
    ]
    assert len(root_runs) == 1
    assert all(
        run["name"] == "rag.request"
        for run in client.created
        if run.get("parent_run_id") is None
    ), [
        (run["name"], run.get("parent_run_id"))
        for run in client.created
    ]
    root = root_runs[0]
    assert root["tags"] == ["01-rag", "interactive"]
    assert root["extra"]["metadata"]["session_id"] == "session-1"
    assert root["extra"]["metadata"]["channel"] == "streamlit"
    assert root["inputs"]["auth_context"]["authorization"] == REDACTED_CREDENTIAL
    assert any(run["name"] == "RunnableLambda" for run in client.created)
    assert not any(
        _contains_marker(payload)
        for payload in [*client.created, *client.updated]
    )

    history = get_history("session-1").messages
    assert [(message.type, message.content) for message in history] == [
        ("human", "保修期多久？"),
        ("ai", "保修期为 12 个月。[S1]"),
    ]


def test_production_history_chain_propagates_business_callback_to_nested_generation(
    monkeypatch,
):
    """Clearing the inherited Runnable config drops nested business callbacks."""
    import rag.chain as chain
    from rag import langsmith_tracing as tracing

    _configure_ambient_tracing(monkeypatch)
    _patch_offline_rag_dependencies(monkeypatch, chain)
    default_client_calls = _configure_client_boundary(
        monkeypatch, tracing, fail=False
    )
    observer = _TracingStateObserver()
    callback_tag = "business-monitoring-tag"
    callback_metadata = {"business_monitor": "answer-generation"}
    invoke_config = {
        "callbacks": [observer],
        "tags": [callback_tag],
        "metadata": callback_metadata,
        "configurable": {"session_id": "callback-propagation-session"},
    }

    history_chain, _ = chain.create_chain_with_history()
    result = history_chain.invoke(
        {"question": "保修期多久？"},
        config=invoke_config,
    )

    assert result["answer"] == "保修期为 12 个月。[S1]"
    nested_prompt_events = [
        event
        for event in observer.chain_start_events
        if event["name"] == "ChatPromptTemplate"
    ]
    assert len(nested_prompt_events) == 1
    assert callback_tag in nested_prompt_events[0]["tags"]
    assert (
        nested_prompt_events[0]["metadata"]["business_monitor"]
        == "answer-generation"
    )
    assert invoke_config == {
        "callbacks": [observer],
        "tags": ["business-monitoring-tag"],
        "metadata": {"business_monitor": "answer-generation"},
        "configurable": {"session_id": "callback-propagation-session"},
    }
    assert default_client_calls == []
    assert len(_TransportRecordingClient.instances) == 1
    client = _TransportRecordingClient.instances[0]
    assert [
        run["name"]
        for run in client.created
        if run.get("parent_run_id") is None
    ] == ["rag.request"]


def test_safe_client_failure_keeps_production_history_execution_untraced(
    monkeypatch,
):
    """Falling back to ambient tracing would leak automatic Runnable traces."""
    import rag.chain as chain
    from rag import langsmith_tracing as tracing

    _configure_ambient_tracing(monkeypatch)
    _patch_offline_rag_dependencies(monkeypatch, chain)
    default_client_calls = _configure_client_boundary(
        monkeypatch, tracing, fail=True
    )
    observer = _TracingStateObserver()

    history_chain, get_history = chain.create_chain_with_history()
    result = history_chain.invoke(
        {
            "question": "保修期多久？",
            "auth_context": {"authorization": f"Bearer {_CREDENTIAL_MARKER}"},
        },
        config={
            "callbacks": [observer],
            "tags": ["01-rag"],
            "metadata": {"session_id": "failed-client-session"},
            "configurable": {"session_id": "failed-client-session"},
        },
    )

    assert result["answer"] == "保修期为 12 个月。[S1]"
    assert observer.chain_start_states
    assert set(observer.chain_start_states) == {False}
    assert default_client_calls == []
    assert _TransportRecordingClient.instances == []
    assert [
        (message.type, message.content)
        for message in get_history("failed-client-session").messages
    ] == [
        ("human", "保修期多久？"),
        ("ai", "保修期为 12 个月。[S1]"),
    ]
