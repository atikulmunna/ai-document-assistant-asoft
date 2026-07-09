"""Central configuration, loaded once from the environment.

Keeping every tunable in one place (single source of truth) means the rest of
the code never reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repository layout anchors.
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "sample_docs"
INDEX_DIR = BASE_DIR / "data" / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def _int_env(name: str, default: int) -> int:
    """Read an int env var, falling back to a default on missing/invalid input."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    generation_model: str
    embedding_model: str
    retrieval_top_k: int
    # Chunks below this similarity are treated as irrelevant, which drives the
    # "insufficient information" (no-hallucination) behaviour.
    min_similarity: float
    max_question_length: int


def load_settings() -> Settings:
    """Build the immutable settings object from the environment."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        generation_model=os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash").strip(),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        retrieval_top_k=_int_env("RETRIEVAL_TOP_K", 5),
        min_similarity=float(os.getenv("RETRIEVAL_MIN_SIMILARITY", "0.55")),
        max_question_length=_int_env("MAX_QUESTION_LENGTH", 1000),
    )


settings = load_settings()
