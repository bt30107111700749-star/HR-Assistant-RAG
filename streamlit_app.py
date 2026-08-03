"""Streamlit interface for the Corporate HR Assistant."""

from __future__ import annotations

import logging
import os
import re
import sys
from importlib import import_module
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Streamlit requires page configuration before any other Streamlit command.
st.set_page_config(
    page_title="Corporate HR Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
load_dotenv(BASE_DIR / ".env")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

doc_loader = import_module("01_documents")
store_module = import_module("05_create_chroma_store")
retriever_module = import_module("06_retrieve_context")
prompting_module = import_module("07_prompting")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("streamlit_app")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
            color: #f1f5f9 !important;
        }
        [data-testid="stHeader"] {
            background: rgba(15, 23, 42, 0.95) !important;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.92) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #94a3b8 !important;
        }
        [data-testid="stMetric"] {
            background: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 14px !important;
            padding: 12px 16px !important;
        }
        [data-testid="stMetricValue"] {
            color: #a5b4fc !important;
            font-weight: 800 !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            font-weight: 600 !important;
        }
        [data-testid="stChatMessage"] {
            background: rgba(30, 41, 59, 0.72) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
        }
        [data-testid="stChatMessage"] * {
            color: #ffffff !important;
            line-height: 1.6 !important;
        }
        [data-testid="stChatInput"] {
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            background-color: rgba(30, 41, 59, 0.95) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "How does the recruitment process work?",
    "What are the appointment policies?",
    "What is the compensation policy?",
    "What professional development opportunities are available?",
    "How can an employee file a complaint?",
    "Who can access employee records?",
    "What corrective actions may be taken?",
]


def resolve_api_key() -> str | None:
    """Resolve the OpenRouter key from Streamlit Secrets or local environment."""
    try:
        secret = st.secrets.get("OPENROUTER_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    value = os.getenv("OPENROUTER_API_KEY", "").strip()
    return value or None


def resolve_model_name() -> str:
    try:
        secret = st.secrets.get("OPENROUTER_MODEL")
        if secret:
            return str(secret).strip().strip('"').strip("'")
    except Exception:
        pass
    return prompting_module.resolve_model_name()


def safe_pdf_name(original_name: str) -> str:
    """Keep uploads inside the documents directory and normalize the filename."""
    base_name = Path(original_name).name
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(base_name).stem).strip(" ._")
    stem = stem or "uploaded_policy"
    return f"{stem}.pdf"


def save_uploaded_pdfs(uploaded_files) -> int:
    saved = 0
    for uploaded_file in uploaded_files or []:
        target = DOCUMENTS_DIR / safe_pdf_name(uploaded_file.name)
        target.write_bytes(uploaded_file.getbuffer())
        saved += 1
    return saved


@st.cache_resource(show_spinner=False)
def load_vector_store(force_rebuild: bool = False):
    return store_module.get_or_create_chroma_store(force_rebuild=force_rebuild)


@st.cache_resource(show_spinner=False)
def load_llm(api_key: str, model_name: str):
    return prompting_module.get_llm(api_key=api_key, model_name=model_name)


def source_to_dict(source) -> dict:
    return {
        "source": source.source,
        "page_number": source.page_number,
        "similarity_score": source.similarity_score,
    }


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("📄 Source Documents"):
        for source in sources:
            st.markdown(
                f"- **{source['source']}** — Page {source['page_number']} "
                f"(relevance: {source['similarity_score']:.2f})"
            )


st.session_state.setdefault("messages", [])
st.session_state.setdefault("rebuild_requested", False)

api_key = resolve_api_key()
model_name = resolve_model_name()
pdf_files = doc_loader.list_pdf_files()
num_files = len(pdf_files)
db_exists = store_module.chroma_store_exists()

with st.sidebar:
    st.markdown("## 🏢 Corporate HR Assistant")
    st.caption("RAG-powered HR policy Q&A — OpenRouter + ChromaDB")
    st.divider()

    st.markdown("### 📥 Policy PDFs")
    uploads = st.file_uploader(
        "Upload one or more HR policy PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Uploaded files are saved to the app's documents folder.",
    )
    if st.button("💾 Save PDFs & Rebuild", use_container_width=True, disabled=not uploads):
        try:
            count = save_uploaded_pdfs(uploads)
            st.session_state.rebuild_requested = True
            load_vector_store.clear()
            st.success(f"Saved {count} PDF file(s).")
            st.rerun()
        except Exception:
            logger.exception("Failed to save uploaded PDFs.")
            st.error("The uploaded PDF files could not be saved.")

    st.divider()
    st.markdown("### 📊 Knowledge Base Status")
    col1, col2 = st.columns(2)
    col1.metric("PDF Files", num_files)
    chunks_metric_slot = col2.empty()
    chunks_metric_slot.metric(
        "Indexed Chunks",
        st.session_state.get("vector_store_stats", {}).get("num_chunks", "—"),
    )
    db_status_slot = st.empty()
    db_status_slot.markdown(
        f"**Vector DB status:** {'🟢 Ready' if db_exists else '🟡 Not built yet'}"
    )

    if api_key:
        st.caption(f"Model: {model_name}")
    else:
        st.error("OPENROUTER_API_KEY is not configured.")

    st.divider()
    st.markdown("### 🎛️ Controls")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🔄 Rebuild Vector Database",
        use_container_width=True,
        disabled=num_files == 0,
    ):
        st.session_state.rebuild_requested = True
        load_vector_store.clear()
        st.rerun()

    st.divider()
    st.markdown("### 💡 Example Questions")
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, key=f"example_{question}", use_container_width=True):
            st.session_state.pending_question = question

st.title("🏢 Corporate HR Assistant")
st.markdown(
    "Ask questions about company HR policies. Answers are generated only from "
    "the PDF documents indexed by this app."
)
st.divider()

if num_files == 0:
    st.warning(
        "No PDF files are available. Upload HR policy PDFs from the sidebar, "
        "then click **Save PDFs & Rebuild**."
    )
    st.stop()

try:
    with st.spinner(
        "🔄 Rebuilding vector database..."
        if st.session_state.rebuild_requested
        else "Loading knowledge base..."
    ):
        vector_store = load_vector_store(
            force_rebuild=st.session_state.rebuild_requested
        )
        st.session_state.vector_store_stats = store_module.get_collection_stats(
            vector_store
        )
        chunks_metric_slot.metric(
            "Indexed Chunks",
            st.session_state.vector_store_stats.get("num_chunks", 0),
        )
        db_status_slot.markdown("**Vector DB status:** 🟢 Ready")
    if st.session_state.rebuild_requested:
        st.session_state.rebuild_requested = False
        st.success("✅ Vector database rebuilt successfully.")
except Exception as exc:
    logger.exception("Failed to load or build the vector store.")
    st.error(f"Could not build the knowledge base: {exc}")
    st.stop()

if not api_key:
    st.warning(
        "The PDFs are indexed, but chat is disabled until OPENROUTER_API_KEY is "
        "added in Streamlit Secrets or a local .env file."
    )
    st.stop()

try:
    llm = load_llm(api_key, model_name)
except Exception as exc:
    logger.exception("Failed to initialize the LLM.")
    st.error(f"Could not initialize OpenRouter: {exc}")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

pending_question = st.session_state.pop("pending_question", None)
user_question = st.chat_input("Ask a question about HR policy...") or pending_question

if user_question:
    user_question = user_question.strip()
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policy documents..."):
            try:
                retrieved_chunks = retriever_module.retrieve_relevant_chunks(
                    vector_store,
                    user_question,
                )
                context = retriever_module.format_context_for_prompt(retrieved_chunks)
                history = [
                    {"role": item["role"], "content": item["content"]}
                    for item in st.session_state.messages[:-1][-6:]
                ]
                result = prompting_module.generate_answer(
                    question=user_question,
                    context=context,
                    llm=llm,
                    chat_history=history,
                )
                sources = (
                    []
                    if result.is_fallback
                    else [source_to_dict(item) for item in retrieved_chunks]
                )
                st.markdown(result.answer)
                render_sources(sources)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result.answer,
                        "sources": sources,
                    }
                )
            except Exception:
                logger.exception("Failed to answer the question.")
                error_message = (
                    "⚠️ The question could not be answered. Check the app logs, "
                    "API key, model access, and vector database."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
