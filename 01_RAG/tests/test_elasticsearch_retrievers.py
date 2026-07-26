from types import SimpleNamespace

from langchain_core.documents import Document


def _config():
    return SimpleNamespace(
        READ_ALIAS="rag-child-chunks-read",
        DENSE_NUM_CANDIDATES=300,
    )


class _SearchClient:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"hits": {"hits": self.hits}}


def _store(client):
    return SimpleNamespace(client=client, config=_config())


def _hit(child_id, score=1.0):
    return {
        "_id": f"storage-{child_id}",
        "_score": score,
        "_source": {
            "content": child_id,
            "storage_id": f"storage-{child_id}",
            "child_id": child_id,
            "parent_id": f"parent-{child_id}",
            "doc_id": "doc-1",
            "doc_version": "v1",
            "status": "active",
        },
    }


def test_bm25_query_pushes_active_and_time_filters():
    from rag.elasticsearch_retrievers import ElasticsearchBM25Retriever

    client = _SearchClient([_hit("bm25")])
    retriever = ElasticsearchBM25Retriever(
        store=_store(client),
        top_k=50,
        metadata_filter={
            "$and": [
                {"has_doc_date": True},
                {"doc_date_min": {"$lte": 20241231}},
                {"doc_date_max": {"$gte": 20240101}},
            ]
        },
    )

    docs = retriever.invoke("保修期")

    call = client.calls[0]
    assert call["size"] == 50
    assert call["query"]["bool"]["must"] == [
        {"match": {"content": {"query": "保修期"}}}
    ]
    assert {"term": {"status": "active"}} in call["query"]["bool"]["filter"]
    assert {"term": {"has_doc_date": True}} in call["query"]["bool"]["filter"]
    assert docs[0].metadata["retrieval_source"] == "bm25"
    assert docs[0].metadata["bm25_score"] == 1.0


def test_dense_knn_uses_same_filters_and_expected_window():
    from rag.elasticsearch_retrievers import ElasticsearchDenseRetriever

    client = _SearchClient([_hit("dense", 0.9)])
    metadata_filter = {"visibility": "public"}

    class FakeEmbeddings:
        def embed_query(self, query):
            assert query == "保修期"
            return [0.1, 0.2, 0.3]

    retriever = ElasticsearchDenseRetriever(
        store=_store(client),
        embeddings=FakeEmbeddings(),
        top_k=50,
        metadata_filter=metadata_filter,
    )

    docs = retriever.invoke("保修期")

    knn = client.calls[0]["knn"]
    assert knn["field"] == "embedding"
    assert knn["k"] == 50
    assert knn["num_candidates"] == 300
    assert knn["filter"] == [
        {"term": {"status": "active"}},
        {"term": {"visibility": "public"}},
    ]
    assert docs[0].metadata["retrieval_source"] == "dense"
    assert docs[0].metadata["dense_score"] == 0.9


def test_bm25_and_dense_build_equivalent_shared_filters():
    from rag.elasticsearch_retrievers import (
        ElasticsearchBM25Retriever,
        ElasticsearchDenseRetriever,
    )

    metadata_filter = {
        "$and": [
            {"visibility": "public"},
            {"doc_date_min": {"$lte": 20241231}},
        ]
    }
    bm25_client = _SearchClient([])
    dense_client = _SearchClient([])

    class FakeEmbeddings:
        def embed_query(self, query):
            return [0.1, 0.2, 0.3]

    ElasticsearchBM25Retriever(
        store=_store(bm25_client),
        top_k=50,
        metadata_filter=metadata_filter,
    ).invoke("q")
    ElasticsearchDenseRetriever(
        store=_store(dense_client),
        embeddings=FakeEmbeddings(),
        top_k=50,
        metadata_filter=metadata_filter,
    ).invoke("q")

    assert (
        bm25_client.calls[0]["query"]["bool"]["filter"]
        == dense_client.calls[0]["knn"]["filter"]
    )


def _doc(child_id, source):
    return Document(
        page_content=child_id,
        metadata={
            "child_id": child_id,
            "parent_id": f"parent-{child_id}",
            "doc_id": "doc-1",
            "retrieval_source": source,
        },
    )


def test_weighted_rrf_fuses_by_child_id_and_is_deterministic():
    from rag.elasticsearch_retrievers import weighted_rrf

    result = weighted_rrf(
        bm25_documents=[_doc("shared", "bm25"), _doc("bm25-only", "bm25")],
        dense_documents=[_doc("dense-only", "dense"), _doc("shared", "dense")],
        bm25_weight=0.4,
        dense_weight=0.6,
        rrf_k=60,
        top_k=80,
    )

    assert [doc.metadata["child_id"] for doc in result] == [
        "shared",
        "dense-only",
        "bm25-only",
    ]
    assert result[0].metadata["retrieval_sources"] == ["bm25", "dense"]
    assert result[0].metadata["rrf_score"] > result[1].metadata["rrf_score"]


def test_hybrid_retriever_degrades_when_one_route_fails():
    from rag.elasticsearch_retrievers import HybridChildRetriever

    class FailingRetriever:
        def invoke(self, query):
            raise RuntimeError("dense failed")

    class WorkingRetriever:
        def invoke(self, query):
            return [_doc("bm25-only", "bm25")]

    hybrid = HybridChildRetriever(
        bm25=WorkingRetriever(),
        dense=FailingRetriever(),
        bm25_weight=0.4,
        dense_weight=0.6,
        rrf_k=60,
        top_k=80,
    )

    result = hybrid.invoke("q")

    assert [doc.metadata["child_id"] for doc in result] == ["bm25-only"]
    assert hybrid.last_errors == {"dense": "dense failed"}


def test_hybrid_retriever_returns_empty_when_both_routes_fail():
    from rag.elasticsearch_retrievers import HybridChildRetriever

    class FailingRetriever:
        def __init__(self, message):
            self.message = message

        def invoke(self, query):
            raise RuntimeError(self.message)

    hybrid = HybridChildRetriever(
        bm25=FailingRetriever("bm25 failed"),
        dense=FailingRetriever("dense failed"),
        bm25_weight=0.4,
        dense_weight=0.6,
        rrf_k=60,
        top_k=80,
    )

    assert hybrid.invoke("q") == []
    assert hybrid.last_errors == {
        "bm25": "bm25 failed",
        "dense": "dense failed",
    }
