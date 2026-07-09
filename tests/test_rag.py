"""Tests for RAG orchestration, with Gemini calls mocked out."""
from __future__ import annotations

import numpy as np
import pytest

from app import gemini
from app.config import Settings
from app.ingest import Chunk
from app.rag import INSUFFICIENT_ANSWER, RagService
from app.vectorstore import VectorStore


def _settings(min_similarity: float = 0.55, top_k: int = 3) -> Settings:
    return Settings(
        gemini_api_key="test",
        generation_model="test-gen",
        embedding_model="test-embed",
        groq_api_key="",
        groq_model="test-groq",
        retrieval_top_k=top_k,
        min_similarity=min_similarity,
        max_question_length=1000,
    )


def _store() -> VectorStore:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    chunks = [
        Chunk(0, "HR Policy.pdf", 1, "Standard working hours are 9:00 AM to 6:00 PM."),
        Chunk(1, "Leave Policy.pdf", 1, "Annual leave is 20 days per year."),
    ]
    return VectorStore(embeddings, chunks)


def test_grounded_answer_returns_citations(monkeypatch):
    monkeypatch.setattr(gemini, "embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(gemini, "generate_answer", lambda q, ctx: "Hours are 9 AM to 6 PM.")

    service = RagService(_store(), _settings())
    resp = service.answer("What are the working hours?")

    assert resp.grounded is True
    assert resp.answer == "Hours are 9 AM to 6 PM."
    assert len(resp.citations) >= 1
    assert resp.citations[0].document == "HR Policy.pdf"
    assert resp.citations[0].page == 1


def test_below_threshold_refuses_without_calling_llm(monkeypatch):
    # Equidistant from both chunks (~0.71 cosine), below the 0.99 floor.
    monkeypatch.setattr(gemini, "embed_query", lambda q: np.array([1.0, 1.0], dtype=np.float32))

    def _fail(*args, **kwargs):  # must never be reached
        raise AssertionError("generate_answer should not be called when nothing is relevant")

    monkeypatch.setattr(gemini, "generate_answer", _fail)

    service = RagService(_store(), _settings(min_similarity=0.99))
    resp = service.answer("What is the airspeed of a swallow?")

    assert resp.grounded is False
    assert resp.answer == INSUFFICIENT_ANSWER
    assert resp.citations == []


def test_model_refusal_is_passed_through(monkeypatch):
    monkeypatch.setattr(gemini, "embed_query", lambda q: np.array([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(gemini, "generate_answer", lambda q, ctx: INSUFFICIENT_ANSWER)

    service = RagService(_store(), _settings())
    resp = service.answer("Something the passages do not cover.")

    assert resp.grounded is False
    assert resp.answer == INSUFFICIENT_ANSWER
    assert resp.citations == []


def test_blank_question_short_circuits(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("no Gemini call expected for blank input")

    monkeypatch.setattr(gemini, "embed_query", _fail)
    service = RagService(_store(), _settings())
    resp = service.answer("   ")

    assert resp.grounded is False
    assert resp.citations == []
