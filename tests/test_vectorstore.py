"""Tests for the NumPy-backed vector store."""
from __future__ import annotations

import numpy as np
import pytest

from app.ingest import Chunk
from app.vectorstore import VectorStore, load, save


def _chunk(i: int) -> Chunk:
    return Chunk(id=i, document=f"doc{i}.pdf", page=i + 1, text=f"text {i}")


def test_search_returns_most_similar_first():
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    store = VectorStore(embeddings, chunks)

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert [c.id for c, _ in results] == [0, 2]
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_search_handles_zero_query_vector():
    store = VectorStore(np.array([[1.0, 0.0]], dtype=np.float32), [_chunk(0)])
    assert store.search(np.array([0.0, 0.0], dtype=np.float32), top_k=1) == []


def test_empty_store_returns_no_results():
    store = VectorStore(np.empty((0, 0), dtype=np.float32), [])
    assert store.size == 0
    assert store.search(np.array([1.0], dtype=np.float32), top_k=3) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        VectorStore(np.array([[1.0, 0.0]], dtype=np.float32), [])


def test_document_count_counts_distinct_documents():
    embeddings = np.eye(3, dtype=np.float32)
    chunks = [Chunk(0, "a.pdf", 1, "x"), Chunk(1, "a.pdf", 2, "y"), Chunk(2, "b.pdf", 1, "z")]
    store = VectorStore(embeddings, chunks)
    assert store.document_count == 2


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    import app.vectorstore as vs

    index_dir = tmp_path / "index"
    monkeypatch.setattr(vs, "INDEX_DIR", index_dir)
    monkeypatch.setattr(vs, "EMBEDDINGS_PATH", index_dir / "embeddings.npy")
    monkeypatch.setattr(vs, "CHUNKS_PATH", index_dir / "chunks.json")

    embeddings = np.array([[0.5, 0.5], [0.1, 0.9]], dtype=np.float32)
    chunks = [_chunk(0), _chunk(1)]
    save(embeddings, chunks)

    loaded = load()
    assert loaded.size == 2
    assert loaded.document_count == 2
    # Metadata survives the roundtrip.
    top = loaded.search(np.array([0.5, 0.5], dtype=np.float32), top_k=1)
    assert top[0][0].text == "text 0"


def test_load_missing_index_raises(tmp_path, monkeypatch):
    import app.vectorstore as vs

    monkeypatch.setattr(vs, "EMBEDDINGS_PATH", tmp_path / "nope.npy")
    monkeypatch.setattr(vs, "CHUNKS_PATH", tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError):
        load()
