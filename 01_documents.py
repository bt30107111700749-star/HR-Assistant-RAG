"""
Document loading for the Corporate HR Assistant.

The module discovers PDF files in ``documents/`` next to this file and loads
one LangChain ``Document`` per PDF page. Source and page metadata are preserved
for retrieval-time citations.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger("01_documents")

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"


class DocumentLoaderError(Exception):
    """Raised when the document corpus cannot be discovered or loaded."""


def list_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return PDF files without raising when the directory is empty/missing."""
    if not documents_dir.exists():
        return []

    return sorted(
        (
            path
            for path in documents_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name.lower(),
    )


def discover_pdf_files(documents_dir: Path = DOCUMENTS_DIR) -> List[Path]:
    """Return all PDF files, or raise a clear error when none are available."""
    pdf_files = list_pdf_files(documents_dir)
    if not pdf_files:
        raise DocumentLoaderError(
            f"No PDF files found in '{documents_dir}'. "
            "Add at least one HR policy PDF before building the vector store."
        )

    logger.info("Discovered %d PDF file(s) in '%s'.", len(pdf_files), documents_dir)
    return pdf_files


def get_documents_fingerprint(documents_dir: Path = DOCUMENTS_DIR) -> str:
    """Create a stable fingerprint from PDF names, sizes, and content bytes."""
    digest = hashlib.sha256()
    for pdf_path in list_pdf_files(documents_dir):
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(str(pdf_path.stat().st_size).encode("ascii"))
        with pdf_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_single_pdf(pdf_path: Path) -> List[Document]:
    """Load one PDF into page-level LangChain documents."""
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:
        logger.exception("Failed to load '%s'.", pdf_path.name)
        raise DocumentLoaderError(f"Failed to load '{pdf_path.name}': {exc}") from exc

    if not pages:
        raise DocumentLoaderError(f"'{pdf_path.name}' contains no readable pages.")

    total_pages = len(pages)
    for page in pages:
        zero_indexed_page = int(page.metadata.get("page", 0))
        page.metadata.update(
            {
                "source": pdf_path.name,
                "file_path": str(pdf_path),
                "page": zero_indexed_page,
                "page_number": zero_indexed_page + 1,
                "total_pages": total_pages,
            }
        )

    logger.info("Loaded '%s' -> %d page(s).", pdf_path.name, total_pages)
    return pages


def load_all_documents(documents_dir: Path = DOCUMENTS_DIR) -> List[Document]:
    """Load all readable PDF pages from the document corpus."""
    pdf_files = discover_pdf_files(documents_dir)
    all_documents: List[Document] = []
    failed_files: List[str] = []

    for pdf_path in pdf_files:
        try:
            all_documents.extend(load_single_pdf(pdf_path))
        except DocumentLoaderError:
            failed_files.append(pdf_path.name)

    if failed_files:
        logger.warning("Skipped unreadable PDF(s): %s", ", ".join(failed_files))

    if not all_documents:
        raise DocumentLoaderError(
            "No readable PDF pages were loaded. Check the files in the documents folder."
        )

    logger.info(
        "Document loading complete: %d PDF(s), %d page(s).",
        len(pdf_files) - len(failed_files),
        len(all_documents),
    )
    return all_documents


def get_document_summary(documents: List[Document]) -> dict:
    """Return basic corpus statistics for the UI."""
    unique_sources = sorted(
        {doc.metadata.get("source", "unknown") for doc in documents}
    )
    return {
        "num_files": len(unique_sources),
        "num_pages": len(documents),
        "files": unique_sources,
    }


if __name__ == "__main__":
    try:
        docs = load_all_documents()
        print(get_document_summary(docs))
    except DocumentLoaderError as exc:
        logger.error(str(exc))
