"""
02_preprocessing.py
====================
Text Preprocessing Module — Corporate HR Assistant using RAG

Responsibility (Step 2 of the pipeline):
    Document Loading -> [PREPROCESSING] -> Chunking -> ...

PDF text extraction is messy: extra whitespace, broken line breaks from
page layout, repeated headers/footers, and stray control characters.
This module cleans the raw page text produced by 01_documents.py while
leaving metadata (source, page_number, etc.) completely untouched, since
that metadata is required later for citations.
"""

import re
import logging
from typing import List

from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("02_preprocessing")


def clean_text(raw_text: str) -> str:
    """
    Clean a single block of raw PDF-extracted text.

    Steps:
        1. Normalize all whitespace runs (spaces, tabs) to a single space.
        2. Collapse 3+ consecutive newlines down to a double newline
           (keeps paragraph breaks, removes excessive blank space).
        3. Strip stray non-printable / control characters.
        4. Trim leading/trailing whitespace.

    Args:
        raw_text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text string.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove non-printable / control characters (keep newlines and tabs
    # for now, they are normalized below).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse horizontal whitespace (spaces/tabs) runs into a single space.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines (with optional whitespace between) into exactly
    # two newlines, preserving paragraph structure.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Remove stray spaces right before/after newlines.
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def is_low_content_page(text: str, min_chars: int = 20) -> bool:
    """
    Detect near-empty pages (e.g. a lone signature page, a blank divider
    page) that add no retrieval value and would only dilute the vector
    store with noise.

    Args:
        text: Cleaned page text.
        min_chars: Minimum character count to be considered meaningful.

    Returns:
        True if the page should be treated as low-content / skippable.
    """
    return len(text.strip()) < min_chars


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Clean the page_content of every Document while preserving metadata.

    Documents that end up empty or near-empty after cleaning are dropped,
    since they carry no useful information for retrieval and would only
    add noise / wasted embeddings.

    Args:
        documents: List of raw Document objects from 01_documents.py.

    Returns:
        List of Document objects with cleaned page_content. Metadata
        (source, page, page_number, total_pages, file_path) is preserved
        exactly as-is.
    """
    cleaned_documents: List[Document] = []
    skipped_count = 0

    for doc in documents:
        cleaned = clean_text(doc.page_content)

        if is_low_content_page(cleaned):
            skipped_count += 1
            logger.debug(
                "Skipping low-content page: %s (page %s)",
                doc.metadata.get("source", "unknown"),
                doc.metadata.get("page_number", "?"),
            )
            continue

        # Preserve all original metadata; only page_content changes.
        cleaned_documents.append(
            Document(page_content=cleaned, metadata=dict(doc.metadata))
        )

    logger.info(
        "Preprocessing complete: %d page(s) cleaned, %d low-content page(s) skipped.",
        len(cleaned_documents), skipped_count,
    )

    return cleaned_documents


# --------------------------------------------------------------------------
# Manual test entry point:
#   python 02_preprocessing.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from importlib import import_module
    import sys
    sys.path.insert(0, ".")
    loader = import_module("01_documents")

    raw_docs = loader.load_all_documents()
    clean_docs = preprocess_documents(raw_docs)

    print("\n=== Preprocessing Summary ===")
    print(f"Raw pages       : {len(raw_docs)}")
    print(f"Cleaned pages   : {len(clean_docs)}")

    print("\n=== Sample Cleaned Document ===")
    sample = clean_docs[0]
    print(f"Source      : {sample.metadata['source']}")
    print(f"Page number : {sample.metadata['page_number']}")
    print(f"Content preview:\n{sample.page_content[:300]}...")
