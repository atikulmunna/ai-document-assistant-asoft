"""Pydantic models that define the public API contract."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000,
                          description="Natural-language question about the corpus.")


class Citation(BaseModel):
    document: str = Field(..., description="Source document file name.")
    page: int = Field(..., description="1-based page number within the document.")
    snippet: str = Field(..., description="Excerpt of the retrieved passage.")
    score: float = Field(..., description="Cosine similarity to the question (0 to 1).")


class QueryResponse(BaseModel):
    answer: str
    # Empty when the corpus does not contain enough information to answer.
    citations: list[Citation]
    grounded: bool = Field(
        ...,
        description="True when the answer is supported by retrieved passages.",
    )


class HealthResponse(BaseModel):
    status: str
    documents: int
    chunks: int
