from pathlib import Path
import sys

from langchain_core.documents import Document


sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rerank_config_enabled_by_default_for_streamlit_app(monkeypatch):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)

    from config import RerankConfig

    assert RerankConfig.ENABLED is True
    assert RerankConfig.TOP_N == 4
    assert RerankConfig.MODEL


def test_rerank_config_can_be_disabled_by_environment(monkeypatch):
    import importlib

    import config

    monkeypatch.setenv("RERANK_ENABLED", "false")

    reloaded_config = importlib.reload(config)

    assert reloaded_config.RerankConfig.ENABLED is False


def test_rerank_documents_disabled_preserves_order_and_does_not_score(monkeypatch):
    from rag import reranker

    docs = [
        Document(page_content="first", metadata={"source": "a.pdf"}),
        Document(page_content="second", metadata={"source": "b.pdf"}),
    ]
    scorer_called = False

    def scorer(query, input_docs):
        nonlocal scorer_called
        scorer_called = True
        return [0.1, 0.9]

    monkeypatch.setattr(reranker.rerank_config, "ENABLED", False)

    result = reranker.rerank_documents("query", docs, scorer=scorer)

    assert result == docs
    assert scorer_called is False


def test_rerank_documents_enabled_sorts_by_score_and_preserves_metadata(monkeypatch):
    from rag import reranker

    docs = [
        Document(page_content="middle", metadata={"source": "mid.pdf"}),
        Document(page_content="best", metadata={"source": "best.pdf"}),
        Document(page_content="low", metadata={"source": "low.pdf"}),
    ]

    monkeypatch.setattr(reranker.rerank_config, "ENABLED", True)
    monkeypatch.setattr(reranker.rerank_config, "TOP_N", 2)

    result = reranker.rerank_documents(
        "query",
        docs,
        scorer=lambda query, input_docs: [0.5, 0.95, 0.1],
    )

    assert [doc.page_content for doc in result] == ["best", "middle"]
    assert result[0].metadata["source"] == "best.pdf"
    assert result[0].metadata["rerank_score"] == 0.95
