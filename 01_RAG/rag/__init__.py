"""
rag/__init__.py
"""
from rag.loader import load_pdf, load_documents_from_dir, get_doc_metadata
from rag.chunker import chunk_documents
from rag.embedder import get_embeddings
from rag.vectorstore import (
    get_vectorstore, add_documents, delete_document,
    list_documents, get_collection_stats, similarity_search_with_threshold
)
from rag.retriever import build_hybrid_retriever, retrieve_with_hybrid
from rag.chain import create_rag_chain, create_chain_with_history

__all__ = [
    "load_pdf",
    "load_documents_from_dir",
    "get_doc_metadata",
    "chunk_documents",
    "get_embeddings",
    "get_vectorstore",
    "add_documents",
    "delete_document",
    "list_documents",
    "get_collection_stats",
    "similarity_search_with_threshold",
    "build_hybrid_retriever",
    "retrieve_with_hybrid",
    "create_rag_chain",
    "create_chain_with_history",
]
