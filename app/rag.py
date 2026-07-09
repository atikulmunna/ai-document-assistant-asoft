"""Retrieval-augmented generation orchestration.

Flow: embed question -> retrieve nearest chunks -> if nothing clears the
similarity floor, refuse honestly without calling the LLM -> otherwise generate
an answer grounded in those chunks and return them as citations.
"""
from __future__ import annotations

from app import gemini
from app.config import Settings
from app.ingest import Chunk
from app.schemas import Citation, QueryResponse
from app.vectorstore import VectorStore

INSUFFICIENT_ANSWER = (
    "I don't have enough information in the provided documents to answer that."
)

_SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    text = text.strip()
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[:_SNIPPET_CHARS].rstrip() + "..."


def _context_block(index: int, chunk: Chunk) -> str:
    return f"[{index}] ({chunk.document}, page {chunk.page}): {chunk.text}"


class RagService:
    """Answers questions against a loaded vector store."""

    def __init__(self, store: VectorStore, settings: Settings):
        self._store = store
        self._settings = settings

    @property
    def store(self) -> VectorStore:
        return self._store

    def answer(self, question: str) -> QueryResponse:
        question = question.strip()
        if not question:
            # Defensive: the API schema already rejects empty input.
            return QueryResponse(answer=INSUFFICIENT_ANSWER, citations=[], grounded=False)

        query_vector = gemini.embed_query(question)
        hits = self._store.search(query_vector, self._settings.retrieval_top_k)
        relevant = [(c, s) for c, s in hits if s >= self._settings.min_similarity]

        if not relevant:
            return QueryResponse(answer=INSUFFICIENT_ANSWER, citations=[], grounded=False)

        context_blocks = [_context_block(i, chunk) for i, (chunk, _) in enumerate(relevant, 1)]
        answer = gemini.generate_answer(question, context_blocks)

        # The model may still judge the passages insufficient; honour that.
        if answer.strip().rstrip(".") == INSUFFICIENT_ANSWER.rstrip("."):
            return QueryResponse(answer=INSUFFICIENT_ANSWER, citations=[], grounded=False)

        citations = [
            Citation(document=chunk.document, page=chunk.page,
                     snippet=_snippet(chunk.text), score=round(score, 4))
            for chunk, score in relevant
        ]
        return QueryResponse(answer=answer, citations=citations, grounded=True)
