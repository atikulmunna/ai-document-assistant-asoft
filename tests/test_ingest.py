"""Tests for PDF-agnostic chunking logic in app.ingest."""
from __future__ import annotations

import json

import pytest

from app.ingest import (
    MAX_CHUNK_CHARS,
    Chunk,
    _normalize,
    _split_page,
    build_chunks,
    chunk_records,
    iter_ocr_pages,
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
    # The text-bearing handbook is represented (the scanned Labour Act PDF has
    # no text layer, so it enters the corpus via OCR, not build_chunks).
    documents = {c.document for c in chunks}
    assert "Partex-Star-Group.pdf" in documents


def test_build_chunks_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path)


def test_chunk_records_numbers_ids_contiguously_across_sources():
    records = [
        ("Partex-Star-Group.pdf", 4, "Annual leave is thirty days."),
        ("A Handbook on the Bangladesh Labour Act 2006.pdf", 56, "Eight hours a day."),
    ]
    chunks = chunk_records(records)
    assert [c.id for c in chunks] == list(range(len(chunks)))
    docs = {(c.document, c.page) for c in chunks}
    assert ("Partex-Star-Group.pdf", 4) in docs
    assert ("A Handbook on the Bangladesh Labour Act 2006.pdf", 56) in docs


def test_iter_ocr_pages_reads_sidecar_and_skips_empty(tmp_path):
    ocr_file = tmp_path / "labour_act.json"
    ocr_file.write_text(json.dumps({
        "document": "A Handbook on the Bangladesh Labour Act 2006.pdf",
        "pages": [
            {"page": 25, "text": "CHAPTER II Conditions of Service."},
            {"page": 26, "text": "   "},  # blank pages are dropped
        ],
    }), encoding="utf-8")

    records = list(iter_ocr_pages(ocr_file))
    assert records == [("A Handbook on the Bangladesh Labour Act 2006.pdf", 25,
                        "CHAPTER II Conditions of Service.")]


def test_iter_ocr_pages_missing_file_yields_nothing(tmp_path):
    assert list(iter_ocr_pages(tmp_path / "does_not_exist.json")) == []
