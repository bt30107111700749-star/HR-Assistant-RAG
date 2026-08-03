"""Strict grounded prompting through OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("07_prompting")

DEFAULT_MODEL_NAME = "qwen/qwen3-8b"
FALLBACK_ANSWER = "I couldn't find this information in the HR policy documents."

SYSTEM_PROMPT = f"""
You are a Corporate HR Assistant.

Use only the HR policy evidence in the current user's [Source ...] blocks.
Conversation history is context for phrasing only and is never evidence.
Do not use outside knowledge, assumptions, or invented policy details.
When evidence supports the answer, answer clearly and concisely.
When the evidence does not support the answer, output exactly this sentence:

{FALLBACK_ANSWER}
""".strip()


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    is_fallback: bool


def resolve_model_name(model_name: Optional[str] = None) -> str:
    """Resolve a clean OpenRouter model slug from argument or environment."""
    value = model_name or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL_NAME
    return value.strip().strip('"').strip("'")


def build_user_prompt(question: str, context: str) -> str:
    """Assemble the grounded user prompt."""
    return f"""
HR policy evidence:
{context}

Question:
{question.strip()}

Answer using only the HR policy evidence above.
""".strip()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Create the OpenRouter client."""
    resolved_api_key = (
        api_key
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not resolved_api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    return ChatOpenAI(
        model=resolve_model_name(model_name),
        temperature=temperature,
        api_key=resolved_api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
            "X-Title": "Corporate HR Assistant",
        },
        timeout=60,
        max_retries=2,
    )


def generate_answer(
    question: str,
    context: str,
    llm: Optional[ChatOpenAI] = None,
    chat_history: Optional[List[dict]] = None,
) -> RAGResponse:
    """Generate an answer, or return the deterministic fallback without an API call."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not context or not context.strip():
        return RAGResponse(answer=FALLBACK_ANSWER, is_fallback=True)

    llm = llm or get_llm()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for turn in (chat_history or [])[-6:]:
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if turn.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif turn.get("role") == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=build_user_prompt(question, context)))

    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.exception("OpenRouter invocation failed.")
        raise RuntimeError("The language model request failed.") from exc

    answer_text = str(response.content).strip()
    if not answer_text:
        answer_text = FALLBACK_ANSWER

    is_fallback = answer_text.rstrip(".").strip() == FALLBACK_ANSWER.rstrip(".")
    return RAGResponse(answer=answer_text, is_fallback=is_fallback)
