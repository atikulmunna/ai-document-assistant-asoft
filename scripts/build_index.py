"""Build the vector index from the PDF corpus.

Run once whenever the documents change:
    python -m scripts.build_index

Output (data/index/) is baked into the Docker image so the running service
never re-embeds the corpus at startup.
"""
from __future__ import annotations

from app import vectorstore
from app.config import DOCS_DIR, INDEX_DIR, OCR_PATH
from app.gemini import embed_texts
from app.ingest import chunk_records, iter_ocr_pages, iter_pages


def main() -> None:
    print(f"Parsing PDFs in {DOCS_DIR} ...")
    # Text-bearing PDFs plus OCR'd pages from scanned PDFs share one chunker so
    # ids stay contiguous across both sources.
    records = list(iter_pages(DOCS_DIR)) + list(iter_ocr_pages(OCR_PATH))
    chunks = chunk_records(records)
    if not chunks:
        raise SystemExit("No text found. Run scripts/ocr_labour_act.py first if PDFs are scanned.")
    documents = sorted({c.document for c in chunks})
    print(f"Built {len(chunks)} chunks across {len(documents)} documents.")

    print("Embedding chunks with Gemini (RETRIEVAL_DOCUMENT) ...")
    embeddings = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    vectorstore.save(embeddings, chunks)
    print(f"Saved index ({embeddings.shape[0]} vectors, dim {embeddings.shape[1]}) to {INDEX_DIR}.")


if __name__ == "__main__":
    main()
