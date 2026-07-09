"""Build the vector index from the PDF corpus.

Run once whenever the documents change:
    python -m scripts.build_index

Output (data/index/) is baked into the Docker image so the running service
never re-embeds the corpus at startup.
"""
from __future__ import annotations

from app import vectorstore
from app.config import DOCS_DIR, INDEX_DIR
from app.gemini import embed_texts
from app.ingest import build_chunks


def main() -> None:
    print(f"Parsing PDFs in {DOCS_DIR} ...")
    chunks = build_chunks(DOCS_DIR)
    documents = sorted({c.document for c in chunks})
    print(f"Built {len(chunks)} chunks across {len(documents)} documents.")

    print("Embedding chunks with Gemini (RETRIEVAL_DOCUMENT) ...")
    embeddings = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    vectorstore.save(embeddings, chunks)
    print(f"Saved index ({embeddings.shape[0]} vectors, dim {embeddings.shape[1]}) to {INDEX_DIR}.")


if __name__ == "__main__":
    main()
