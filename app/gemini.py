"""Thin wrapper around the Google Gemini SDK for embeddings and generation.

Isolating every Gemini call here keeps the provider swappable and gives the
rest of the app a single, small surface to depend on.
"""
from __future__ import annotations

import logging

import numpy as np
from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger("agamisoft")

# Groq's OpenAI-compatible endpoint (used only as a generation fallback).
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Batching keeps build-time embedding requests within provider limits.
_EMBED_BATCH_SIZE = 100

_client: genai.Client | None = None
_groq_client = None  # lazily created openai.OpenAI pointed at Groq


class GeminiError(RuntimeError):
    """Raised when a generation or embedding call fails or returns nothing usable."""


def _get_client() -> genai.Client:
    """Lazily create a single shared client (fails fast if the key is missing)."""
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise GeminiError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _embed_request(texts: list[str], task_type: str) -> list[list[float]]:
    client = _get_client()
    response = client.models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in response.embeddings]


def embed_texts(texts: list[str], *, task_type: str) -> np.ndarray:
    """Embed a list of texts, returning an (n, dim) float32 array.

    task_type is RETRIEVAL_DOCUMENT for corpus chunks and RETRIEVAL_QUERY for a
    user question; the asymmetry improves retrieval quality.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    try:
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start:start + _EMBED_BATCH_SIZE]
            try:
                vectors.extend(_embed_request(batch, task_type))
            except Exception:
                # Fall back to per-item embedding if batching is rejected.
                for text in batch:
                    vectors.extend(_embed_request([text], task_type))
    except Exception as exc:  # network / auth / quota errors
        raise GeminiError(f"Embedding request failed: {exc}") from exc

    return np.asarray(vectors, dtype=np.float32)


def embed_query(question: str) -> np.ndarray:
    """Embed a single question, returning a 1-D float32 vector."""
    return embed_texts([question], task_type="RETRIEVAL_QUERY")[0]


_SYSTEM_INSTRUCTION = (
    "You are AgamiSoft's internal document assistant. Answer strictly and only "
    "from the numbered context passages provided. Follow these rules:\n"
    "1. If the context does not contain enough information to answer, reply "
    "exactly: \"I don't have enough information in the provided documents to "
    "answer that.\" Do not guess or use outside knowledge.\n"
    "2. Keep answers concise and factual, and prefer the wording of the source.\n"
    "3. Never invent policies, numbers, names, or contacts."
)


def _build_prompt(question: str, context_blocks: list[str]) -> str:
    context = "\n\n".join(context_blocks)
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def _generate_gemini(prompt: str) -> str:
    response = _get_client().models.generate_content(
        model=settings.generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=800,
            # Grounded lookup needs no extended reasoning; disabling thinking
            # keeps answers fast (about 1s vs 30s+ with thinking on).
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


def _get_groq_client():
    """Lazily create the Groq client (OpenAI-compatible) for fallback generation."""
    global _groq_client
    if _groq_client is None:
        from openai import OpenAI  # imported here so the dep is only needed for fallback

        _groq_client = OpenAI(api_key=settings.groq_api_key, base_url=_GROQ_BASE_URL)
    return _groq_client


def _generate_groq(prompt: str) -> str:
    response = _get_groq_client().chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    return (response.choices[0].message.content or "").strip()


def generate_answer(question: str, context_blocks: list[str]) -> str:
    """Generate a grounded answer, using Gemini first and Groq as a fallback.

    Gemini's free tier rate-limits generation, so when it fails (commonly a 429)
    we retry once with Groq. Embeddings and retrieval are unchanged; only the
    final answer is produced by the fallback provider.
    """
    prompt = _build_prompt(question, context_blocks)

    try:
        answer = _generate_gemini(prompt)
        if answer:
            return answer
        raise RuntimeError("empty response")
    except Exception as gemini_exc:
        if not settings.groq_api_key:
            raise GeminiError(f"Gemini generation failed and no fallback is configured: {gemini_exc}")
        logger.warning("Gemini generation failed (%s); falling back to Groq.", gemini_exc)

    try:
        answer = _generate_groq(prompt)
    except Exception as groq_exc:
        raise GeminiError(f"All generation providers failed. Groq: {groq_exc}")
    if not answer:
        raise GeminiError("Groq generation returned an empty response.")
    logger.info("Answer served by Groq fallback (%s).", settings.groq_model)
    return answer
