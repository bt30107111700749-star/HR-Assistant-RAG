"""Retrieve grounded context chunks from Chroma."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma

logger = logging.getLogger("06_retrieve_context")

TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.30"))


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    page_number: int
    chunk_id: str
    similarity_score: float


def retrieve_relevant_chunks(
    vector_store: Chroma,
    query: str,
    top_k: int = TOP_K,
) -> List[RetrievedChunk]:
    """Return relevant chunks above the configured relevance threshold."""
    normalized_query = (query or "").strip()
    if not normalized_query:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query=normalized_query,
            k=top_k,
        )
    except Exception as exc:
        logger.exception("Similarity search failed.")
        raise RuntimeError("The vector database search failed.") from exc

    chunks: List[RetrievedChunk] = []
    for doc, score in results:
        relevance_score = float(score)
        if relevance_score < MIN_RELEVANCE_SCORE:
            continue

        page_number = doc.metadata.get("page_number", -1)
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = -1

        chunks.append(
            RetrievedChunk(
                content=doc.page_content,
                source=str(doc.metadata.get("source", "unknown")),
                page_number=page_number,
                chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
                similarity_score=relevance_score,
            )
        )

    logger.info("Retrieved %d chunk(s) for query '%s'.", len(chunks), normalized_query[:80])
    return chunks


def format_context_for_prompt(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks with source and page labels."""
    blocks = [
        (
            f"[Source {index} | Document: {chunk.source} | Page: {chunk.page_number}]\n"
            f"{chunk.content}"
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)
