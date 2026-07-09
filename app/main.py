"""FastAPI application: query API, health check, and a minimal static UI."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.gemini import GeminiError
from app.rag import RagService
from app.schemas import HealthResponse, QueryRequest, QueryResponse
from app.vectorstore import load as load_store

logger = logging.getLogger("agamisoft")
logging.basicConfig(level=logging.INFO)

# Quiet noisy third-party loggers so our own request logs stand out.
for _noisy in ("httpx", "httpcore", "google_genai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _truncate(text: str, limit: int = 80) -> str:
    """Shorten a question for logging so we never dump huge inputs."""
    return text if len(text) <= limit else text[:limit] + "..."

# Holds the single RagService instance once the index is loaded.
_state: dict[str, RagService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector index once at startup so requests stay fast."""
    try:
        service = RagService(load_store(), settings)
        _state["service"] = service
        logger.info("Loaded index: %d chunks across %d documents.",
                    service.store.size, service.store.document_count)
    except Exception as exc:
        # Start anyway so /health is reachable; /query will report the problem.
        logger.error("Failed to load index at startup: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="AgamiSoft AI Document Assistant",
    description="Ask natural-language questions grounded in the AgamiSoft document corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_service() -> RagService:
    service = _state.get("service")
    if service is None:
        raise HTTPException(status_code=503,
                            detail="Knowledge base is not available. The index failed to load.")
    return service


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    service = _state.get("service")
    if service is None:
        return HealthResponse(status="degraded", documents=0, chunks=0)
    return HealthResponse(status="ok",
                          documents=service.store.document_count,
                          chunks=service.store.size)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if len(question) > settings.max_question_length:
        raise HTTPException(status_code=422,
                            detail=f"Question exceeds {settings.max_question_length} characters.")

    service = _require_service()
    start = time.perf_counter()
    try:
        response = service.answer(question)
    except GeminiError as exc:
        logger.error("Gemini call failed after %.0fms | q=%r",
                     (time.perf_counter() - start) * 1000, _truncate(question))
        raise HTTPException(status_code=502,
                            detail="The AI service is temporarily unavailable. Please try again.")

    elapsed_ms = (time.perf_counter() - start) * 1000
    top_score = f"{response.citations[0].score:.3f}" if response.citations else "n/a"
    logger.info("query answered | grounded=%s | citations=%d | top_score=%s | %.0fms | q=%r",
                response.grounded, len(response.citations), top_score, elapsed_ms,
                _truncate(question))
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Serve JS/CSS assets if any are added later.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
