# 01_RAG Milvus Lite Vector Store Design

## Goal

Replace the 01_RAG child vector store backend from ChromaDB to Milvus Lite local file mode while preserving the public RAG API and existing retrieval flow.

## Scope

- Use `langchain-milvus` as the LangChain vector store integration.
- Use Milvus Lite by default with a local file at `01_RAG/data/milvus.db`.
- Keep the existing embedding responsibility in `rag/embedder.py`; Milvus stores and searches vectors, but does not generate embeddings in this iteration.
- Keep the existing retrieval ranking flow: Milvus dense retrieval, BM25 keyword retrieval, LangChain ensemble fusion, and parent hydration.
- Do not add cross-encoder reranking in this iteration. Track it as the next iteration plan only.
- Do not migrate existing Chroma data automatically. Users rebuild the Milvus index by re-uploading or re-indexing documents.

## Architecture

`config.py` exposes a `MilvusConfig` with collection name, URI, consistency, field names, and index/search parameters. `rag/vectorstore.py` keeps the same function names and external behavior as the Chroma implementation but initializes `langchain_milvus.Milvus` with `connection_args={"uri": milvus_config.URI}`.

The metadata API boundary remains Chroma-shaped for the rest of the project. `build_time_filter()` continues to return the existing dictionary form for unit-level compatibility, and `rag/vectorstore.py` translates that subset into Milvus boolean expressions when calling Milvus search APIs.

Milvus Lite runs a local PyMilvus server under the hood. The implementation registers the ORM connection alias used by `langchain-milvus` and explicitly requests text plus known metadata fields from search results so retrieval returns the same `Document` shape as the previous Chroma path.

## Data Flow

1. PDF loader produces page documents.
2. Chunker produces parent and child chunks with metadata.
3. `add_documents()` deletes any existing rows for the `doc_id`, writes parent chunks into SQLite docstore, and writes child chunks into Milvus.
4. Dense retrieval uses Milvus through `as_retriever()` or `similarity_search_with_relevance_scores()`.
5. BM25 still builds from child documents returned by `vs.get(include=["documents", "metadatas"])`.
6. Parent hydration and date-aware sorting remain unchanged.

## Error Handling

Document-level rollback stays in `add_documents()`: if parent or child writes fail, `delete_document()` is called for the same `doc_id`. Delete/list/stats operations keep best-effort behavior so the Streamlit UI does not crash on empty or partially initialized stores.

## Testing

Tests cover:

- Milvus config defaults to a local `data/milvus.db` file and keeps the versioned collection name.
- `get_vectorstore()` initializes LangChain Milvus with the existing embedding function and Milvus Lite connection args.
- Metadata filter translation supports equality, range, and `$and` filters used by date-aware retrieval.
- Existing vectorstore list/stats tests continue to pass with mocked stores.

## Next Iteration

Cross-encoder reranking should be planned as a separate quality iteration after Milvus is stable. That iteration can add an optional rerank stage after hybrid retrieval and before parent hydration or final context formatting, with an environment flag and evaluation coverage.
