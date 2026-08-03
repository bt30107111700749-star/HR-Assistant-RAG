"""Build and load the persisted Chroma vector database."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import chromadb
from langchain_chroma import Chroma

logger = logging.getLogger("05_create_chroma_store")

BASE_DIR = Path(__file__).resolve().parent
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "hr_policy_documents"

doc_loader = importlib.import_module("01_documents")
preprocessor = importlib.import_module("02_preprocessing")
chunker = importlib.import_module("03_chunking")
vector_repr = importlib.import_module("04_vector_representation")


def _client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def chroma_store_exists(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """Return True only when the named collection exists and contains chunks."""
    if not (persist_dir / "chroma.sqlite3").exists():
        return False

    try:
        collection = _client(persist_dir).get_collection(name=collection_name)
        return collection.count() > 0
    except Exception:
        return False


def _stored_fingerprint(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> str | None:
    try:
        metadata = _client(persist_dir).get_collection(name=collection_name).metadata or {}
        value = metadata.get("corpus_fingerprint")
        return str(value) if value else None
    except Exception:
        return None


def build_chroma_store(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Rebuild the vector database from the current PDF corpus."""
    raw_documents = doc_loader.load_all_documents()
    cleaned_documents = preprocessor.preprocess_documents(raw_documents)
    chunks = chunker.chunk_documents(cleaned_documents)

    if not chunks:
        raise ValueError("No usable text chunks were produced from the PDF files.")

    embedding_model = vector_repr.get_embedding_model()
    client = _client(persist_dir)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    fingerprint = doc_loader.get_documents_fingerprint()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name=collection_name,
        collection_metadata={
            "corpus_fingerprint": fingerprint,
            "hnsw:space": "cosine",
        },
    )

    logger.info(
        "Chroma store built at '%s' with %d chunk(s).",
        persist_dir,
        len(chunks),
    )
    return vector_store


def load_chroma_store(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Load an existing non-empty Chroma collection."""
    if not chroma_store_exists(persist_dir, collection_name):
        raise FileNotFoundError("The Chroma collection is missing or empty.")

    return Chroma(
        client=_client(persist_dir),
        collection_name=collection_name,
        embedding_function=vector_repr.get_embedding_model(),
    )


def get_or_create_chroma_store(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> Chroma:
    """Load the index, rebuilding it when missing, stale, or explicitly requested."""
    current_fingerprint = doc_loader.get_documents_fingerprint()
    stored_fingerprint = _stored_fingerprint(persist_dir, collection_name)
    corpus_changed = bool(current_fingerprint) and current_fingerprint != stored_fingerprint

    if force_rebuild or not chroma_store_exists(persist_dir, collection_name) or corpus_changed:
        if corpus_changed:
            logger.info("PDF corpus changed; rebuilding the Chroma store.")
        return build_chroma_store(persist_dir, collection_name)

    return load_chroma_store(persist_dir, collection_name)


def create_vector_store(documents=None):
    """Backward-compatible rebuild helper used by older callers."""
    if documents is not None:
        logger.warning("The 'documents' argument is ignored; the corpus folder is authoritative.")
    vector_store = build_chroma_store()
    return vector_store, vector_store._collection.count()


def get_collection_stats(vector_store=None, *args, **kwargs) -> dict:
    """Return collection statistics without hiding indexing failures."""
    try:
        store = vector_store or load_chroma_store()
        return {"num_chunks": int(store._collection.count())}
    except Exception:
        return {"num_chunks": 0}
