"""Tests for PDF-agnostic chunking logic in app.ingest."""
from __future__ import annotations

import pytest

from app.ingest import (
    CHUNK_OVERLAP_CHARS,
    MAX_CHUNK_CHARS,
    Chunk,
    _normalize,
    _split_page,
    build_chunks,
)


def test_normalize_collapses_whitespace_and_blank_lines():
    raw = "Line   one\t\there\n\n\n\nLine two"
    assert _normalize(raw) == "Line one here\n\nLine two"


def test_split_short_page_returns_single_chunk():
    text = "A short page of text."
    assert _split_page(text) == [text]


def test_split_empty_page_returns_nothing():
    assert _split_page("") == []


def test_split_long_page_respects_max_chunk_size():
    paragraphs = [f"Paragraph number {i} with some filler content." for i in range(80)]
    text = "\n".join(paragraphs)
    chunks = _split_page(text)

    assert len(chunks) > 1
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # Rejoined chunks must still contain the first and last paragraph content.
    joined = " ".join(chunks)
    assert "Paragraph number 0" in joined
    assert "Paragraph number 79" in joined


def test_single_oversized_paragraph_is_hard_split():
    text = "x" * (MAX_CHUNK_CHARS * 2)
    chunks = _split_page(text)
    assert len(chunks) >= 2
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_build_chunks_on_real_corpus_has_page_metadata():
    from app.config import DOCS_DIR

    chunks = build_chunks(DOCS_DIR)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.page >= 1 for c in chunks)
    assert all(c.text.strip() for c in chunks)
    # IDs are contiguous and unique.
    assert [c.id for c in chunks] == list(range(len(chunks)))
    # Every source document is represented.
    documents = {c.document for c in chunks}
    assert "HR Policy.pdf" in documents
    assert "FAQ.pdf" in documents


def test_build_chunks_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path)
