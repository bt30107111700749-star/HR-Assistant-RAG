"""
03_chunking.py
===============
Chunking Module — Corporate HR Assistant using RAG

Responsibility (Step 3 of the pipeline):
    Preprocessing -> [CHUNKING] -> Embedding Generation -> ...

Splits cleaned page-level Documents into smaller overlapping chunks using
RecursiveCharacterTextSplitter. Chunking is necessary because:
    - Embedding models have limited context windows.
    - Smaller chunks give more precise, focused retrieval results.
    - Overlap preserves context that would otherwise be cut at chunk
      boundaries (e.g. a sentence split across two chunks).

Chunk size/overlap were chosen for short, dense HR policy documents
(1-9 pages each): large enough to retain full policy clauses, small
enough for precise retrieval.
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("03_chunking")

# --------------------------------------------------------------------------
# Chunking configuration
# --------------------------------------------------------------------------
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # characters of overlap between consecutive chunks


def get_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter configured for policy documents.

    The separator list prioritizes splitting on paragraph breaks, then
    sentence-like breaks, before falling back to hard character splits.
    This keeps HR policy clauses intact wherever possible.

    Args:
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split cleaned Documents into overlapping chunks.

    Each resulting chunk keeps the parent page's metadata (source,
    page_number, etc.) and gains a `chunk_id` for traceability.

    Args:
        documents: List of cleaned, page-level Document objects.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunked Document objects ready for embedding.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Add a stable, human-readable chunk id per source file for debugging
    # and for de-duplication if the store is rebuilt.
    per_source_counter = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        per_source_counter[source] = per_source_counter.get(source, 0) + 1
        chunk.metadata["chunk_id"] = f"{source}::chunk_{per_source_counter[source]}"

    logger.info(
        "Chunking complete: %d document page(s) split into %d chunk(s) "
        "(chunk_size=%d, overlap=%d).",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )

    return chunks


# --------------------------------------------------------------------------
# Manual test entry point:
#   python 03_chunking.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from importlib import import_module
    import sys
    sys.path.insert(0, ".")
    loader = import_module("01_documents")
    preprocessor = import_module("02_preprocessing")

    raw_docs = loader.load_all_documents()
    clean_docs = preprocessor.preprocess_documents(raw_docs)
    chunks = chunk_documents(clean_docs)

    print("\n=== Chunking Summary ===")
    print(f"Cleaned pages : {len(clean_docs)}")
    print(f"Total chunks  : {len(chunks)}")
    avg_len = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"Avg chunk length (chars): {avg_len:.0f}")

    print("\n=== Sample Chunk ===")
    sample = chunks[0]
    print(f"Chunk ID    : {sample.metadata['chunk_id']}")
    print(f"Source      : {sample.metadata['source']}")
    print(f"Page number : {sample.metadata['page_number']}")
    print(f"Content:\n{sample.page_content}")
