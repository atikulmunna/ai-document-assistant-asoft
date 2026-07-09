"""In-memory vector store backed by a NumPy matrix.

The corpus is a few dozen chunks, so an exact brute-force cosine search is
instant and avoids a heavyweight vector-database dependency. FAISS or a managed
store would be the drop-in replacement if the corpus grew by orders of
magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import CHUNKS_PATH, EMBEDDINGS_PATH, INDEX_DIR
from app.ingest import Chunk


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero for empty vectors
    return (matrix / norms).astype(np.float32)


class VectorStore:
    def __init__(self, embeddings: np.ndarray, chunks: list[Chunk]):
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length.")
        self._embeddings = _normalize_rows(embeddings) if len(embeddings) else embeddings
        self._chunks = chunks

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len({chunk.document for chunk in self._chunks})

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity) pairs, highest score first."""
        if self.size == 0 or top_k <= 0:
            return []
        query = query_vector.astype(np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm
        scores = self._embeddings @ query
        k = min(top_k, self.size)
        # argpartition for the top-k, then sort just those by score descending.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]


def save(embeddings: np.ndarray, chunks: list[Chunk]) -> None:
    """Persist the index (embeddings + chunk metadata) to data/index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings.astype(np.float32))
    payload = [chunk.__dict__ for chunk in chunks]
    CHUNKS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load() -> VectorStore:
    """Load the persisted index built by scripts/build_index.py."""
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Index not found. Run 'python -m scripts.build_index' first "
            f"(expected {EMBEDDINGS_PATH.name} and {CHUNKS_PATH.name} in {INDEX_DIR})."
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    raw = json.loads(Path(CHUNKS_PATH).read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    return VectorStore(embeddings, chunks)
