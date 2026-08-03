"""Embedding model configuration for the Corporate HR Assistant."""

from __future__ import annotations

import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("04_vector_representation")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()

_embedding_model_cache: HuggingFaceEmbeddings | None = None


def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> HuggingFaceEmbeddings:
    """Build or reuse the process-wide embedding model."""
    global _embedding_model_cache

    if _embedding_model_cache is not None:
        return _embedding_model_cache

    logger.info("Loading embedding model '%s'.", model_name)
    _embedding_model_cache = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return _embedding_model_cache


if __name__ == "__main__":
    embedder = get_embedding_model()
    vector = embedder.embed_query("Employees may request annual leave.")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector length: {len(vector)}")
