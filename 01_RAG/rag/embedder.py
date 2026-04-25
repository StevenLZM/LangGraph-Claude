# -*- coding: utf-8 -*-
"""
rag/embedder.py - Embedding model factory
Priority: DashScope > OpenAI > HuggingFace local
"""
from __future__ import annotations
import time
from typing import List

from langchain_core.embeddings import Embeddings
from config import llm_config


def get_embeddings() -> Embeddings:
    if llm_config.has_dashscope():
        return _get_dashscope_embeddings()
    if llm_config.has_openai():
        return _get_openai_embeddings()
    return _get_huggingface_embeddings()


def _get_dashscope_embeddings() -> Embeddings:
    # Use langchain_community DashScopeEmbeddings (official integration)
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
        import os
        os.environ["DASHSCOPE_API_KEY"] = llm_config.DASHSCOPE_API_KEY
        return DashScopeEmbeddings(
            model=llm_config.EMBEDDING_MODEL,  # text-embedding-v3
            dashscope_api_key=llm_config.DASHSCOPE_API_KEY,
        )
    except ImportError:
        raise ImportError("pip install langchain-community")


def _get_openai_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=llm_config.EMBEDDING_MODEL,
        openai_api_key=llm_config.OPENAI_API_KEY,
        chunk_size=100,
        max_retries=3,
    )


def _get_huggingface_embeddings() -> Embeddings:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        raise ImportError("pip install sentence-transformers")


def embed_with_retry(
    embeddings: Embeddings,
    texts: List[str],
    batch_size: int = 25,
    delay: float = 0.3,
) -> List[List[float]]:
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _embed_batch(batch: List[str]) -> List[List[float]]:
        return embeddings.embed_documents(batch)

    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        all_vectors.extend(_embed_batch(batch))
        if i + batch_size < len(texts):
            time.sleep(delay)
    return all_vectors
