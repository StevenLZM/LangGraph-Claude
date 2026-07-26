import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.documents import Document


pytestmark = pytest.mark.es_integration


@pytest.mark.skipif(
    os.getenv("RUN_ES_INTEGRATION") != "1",
    reason="设置 RUN_ES_INTEGRATION=1 才运行真实 Elasticsearch 测试",
)
def test_real_elasticsearch_child_lifecycle_and_hybrid_queries():
    from elasticsearch import Elasticsearch

    from rag.elasticsearch_retrievers import (
        ElasticsearchBM25Retriever,
        ElasticsearchDenseRetriever,
    )
    from rag.elasticsearch_store import ElasticsearchChildStore

    suffix = uuid4().hex
    physical_index = f"rag-test-{suffix}"
    read_alias = f"rag-test-read-{suffix}"
    write_alias = f"rag-test-write-{suffix}"
    config = SimpleNamespace(
        URL="http://127.0.0.1:9200",
        PHYSICAL_INDEX=physical_index,
        READ_ALIAS=read_alias,
        WRITE_ALIAS=write_alias,
        NUMBER_OF_SHARDS=1,
        NUMBER_OF_REPLICAS=0,
        VERIFY_CERTS=False,
        USERNAME="",
        PASSWORD="",
        API_KEY="",
        CA_CERTS="",
        REQUEST_TIMEOUT=30,
        MAX_RETRIES=3,
        BULK_CHUNK_SIZE=100,
        EMBEDDING_DIMS=3,
        DENSE_NUM_CANDIDATES=10,
    )
    client = Elasticsearch(config.URL, verify_certs=False)
    store = ElasticsearchChildStore(client=client, config=config)

    def child(
        *,
        doc_id: str,
        child_id: str,
        content: str,
        date: int,
    ) -> Document:
        return Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "doc_version": "v1",
                "parent_id": f"parent-{doc_id}",
                "child_id": child_id,
                "source": f"{doc_id}.pdf",
                "page_range": "1",
                "has_doc_date": True,
                "doc_date_min": date,
                "doc_date_max": date,
                "visibility": "public",
            },
        )

    try:
        documents = [
            child(
                doc_id="doc-warranty",
                child_id="child-warranty",
                content="产品保修期为十二个月",
                date=20240101,
            ),
            child(
                doc_id="doc-staging",
                child_id="child-staging",
                content="仅用于验证 staging 不可检索",
                date=20240102,
            ),
            child(
                doc_id="doc-other",
                child_id="child-other",
                content="其他产品资料",
                date=20240201,
            ),
        ]
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ]
        assert store.bulk_stage_children(documents, vectors, "integration-run") == 3
        assert store.activate_version(
            "doc-warranty",
            "v1",
            "integration-run",
        ) == 1
        assert store.activate_version(
            "doc-other",
            "v1",
            "integration-run",
        ) == 1

        aliases = client.indices.get_alias(index=physical_index)
        assert read_alias in aliases[physical_index]["aliases"]
        assert write_alias in aliases[physical_index]["aliases"]
        assert store.count_children() == 2

        shared_filter = {
            "$and": [
                {"visibility": "public"},
                {"doc_date_min": {"$lte": 20241231}},
                {"doc_date_max": {"$gte": 20240101}},
            ]
        }
        bm25 = ElasticsearchBM25Retriever(
            store=store,
            top_k=10,
            metadata_filter=shared_filter,
        ).invoke("保修期")

        class FixedEmbeddings:
            def embed_query(self, query):
                return [1.0, 0.0, 0.0]

        dense = ElasticsearchDenseRetriever(
            store=store,
            embeddings=FixedEmbeddings(),
            top_k=10,
            metadata_filter=shared_filter,
        ).invoke("保修期")

        assert [doc.metadata["child_id"] for doc in bm25] == [
            "child-warranty"
        ]
        dense_ids = {doc.metadata["child_id"] for doc in dense}
        assert "child-warranty" in dense_ids
        assert "child-staging" not in dense_ids

        assert store.delete_document("doc-warranty") == 1
        assert ElasticsearchBM25Retriever(
            store=store,
            top_k=10,
        ).invoke("保修期") == []
    finally:
        client.indices.delete(index=physical_index, ignore_unavailable=True)
