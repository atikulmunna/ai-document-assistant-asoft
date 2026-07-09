"""Document ingestion: parse PDFs and split them into page-tagged chunks.

Chunking keeps the source document and 1-based page number on every chunk so
that answers can cite exactly where the information came from.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

# Chunk sizing (characters). The corpus is tiny and prose-heavy, so modest
# chunks with a little overlap keep each section retrievable on its own.
MAX_CHUNK_CHARS = 900
CHUNK_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class Chunk:
    id: int
    document: str
    page: int
    text: str


def _normalize(text: str) -> str:
    """Collapse runs of whitespace while preserving paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs, then trim each line.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Squeeze 3+ newlines down to a paragraph break.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_page(text: str) -> list[str]:
    """Split one page into overlapping chunks on paragraph boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text] if text else []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}".strip() if current else para
        if len(candidate) <= MAX_CHUNK_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # Start the next chunk with a tail of the previous one for context.
        if chunks and CHUNK_OVERLAP_CHARS:
            tail = chunks[-1][-CHUNK_OVERLAP_CHARS:]
            current = f"{tail}\n{para}".strip()
        else:
            current = para
        # A single oversized paragraph is hard-split to stay within bounds.
        while len(current) > MAX_CHUNK_CHARS:
            chunks.append(current[:MAX_CHUNK_CHARS])
            current = current[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS:]
    if current:
        chunks.append(current)
    return chunks


def iter_pages(docs_dir: Path):
    """Yield (document_name, page_number, page_text) for every PDF page."""
    pdf_paths = sorted(Path(docs_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")
    for path in pdf_paths:
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = _normalize(page.extract_text() or "")
            if text:
                yield path.name, page_number, text


def build_chunks(docs_dir: Path) -> list[Chunk]:
    """Parse every PDF in docs_dir into an ordered list of chunks."""
    chunks: list[Chunk] = []
    for document, page, text in iter_pages(docs_dir):
        for piece in _split_page(text):
            chunks.append(Chunk(id=len(chunks), document=document, page=page, text=piece))
    if not chunks:
        raise ValueError(f"No extractable text found in PDFs under {docs_dir}")
    return chunks


def chunk_to_dict(chunk: Chunk) -> dict:
    return asdict(chunk)
