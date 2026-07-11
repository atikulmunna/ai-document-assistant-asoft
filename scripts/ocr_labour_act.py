"""One-time OCR of the Bangladesh Labour Act handbook's focus chapters.

The handbook is a scanned PDF with no text layer. To keep the repository lean,
only the 13 pages actually used are committed, as a trimmed excerpt in
sample_docs. This script transcribes those pages with Gemini vision and caches
the result to data/ocr/labour_act.json. Both the excerpt and the cache are
committed, so the vector-index build never repeats these vision calls.

The excerpt holds the focus chapters in order (printed page numbers):
    Chapter 02, Conditions of Service and Employment: 25 to 32  (excerpt pages 1 to 8)
    Chapter 09, Working Hours and Leave:              56 to 60  (excerpt pages 9 to 13)

Citations use the handbook's printed page numbers so they match the physical
book. Run once (safe to re-run: pages already cached are skipped, so a
rate-limit stall can be resumed later):
    python -m scripts.ocr_labour_act
"""
from __future__ import annotations

import io
import json
import time

from PIL import Image
from pypdf import PdfReader

from app.config import DOCS_DIR, OCR_PATH
from app.gemini import ocr_image

SOURCE_PDF = "Bangladesh Labour Act 2006 - Chapters 2 and 9 (excerpt).pdf"
DOCUMENT = "A Handbook on the Bangladesh Labour Act 2006.pdf"  # citation label
# Printed page numbers, in the same order as the pages of the excerpt PDF.
FOCUS_PAGES = list(range(25, 33)) + list(range(56, 61))
MAX_IMAGE_SIDE = 2200  # downscale scans to keep the vision request lean
DELAY_SECONDS = 4.0  # gentle spacing to stay under per-minute rate limits
MAX_RETRIES = 5  # transient 503 "high demand" spikes are common; ride them out
RETRY_BACKOFF_SECONDS = 15.0


def _page_image_jpeg(reader: PdfReader, excerpt_index: int) -> bytes:
    """Extract an excerpt page's embedded scan as RGB JPEG bytes."""
    images = list(reader.pages[excerpt_index].images)
    if not images:
        raise RuntimeError(f"No image found on excerpt page {excerpt_index + 1}.")
    image = Image.open(io.BytesIO(images[0].data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _ocr_with_retry(image_bytes: bytes, printed_page: int) -> str:
    """OCR one page, retrying transient failures (e.g. 503 high-demand spikes)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = ocr_image(image_bytes)
            if text.strip():
                return text
            reason = "empty response"
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            reason = str(exc)
        if attempt == MAX_RETRIES:
            raise RuntimeError(f"OCR returned empty text for printed page {printed_page}.")
        wait = RETRY_BACKOFF_SECONDS * attempt
        print(f"  page {printed_page}: attempt {attempt} failed ({reason}); retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_cache() -> dict[int, str]:
    """Load already-transcribed pages so the script can resume after a stall."""
    if not OCR_PATH.exists():
        return {}
    data = json.loads(OCR_PATH.read_text(encoding="utf-8"))
    return {int(entry["page"]): entry["text"] for entry in data.get("pages", [])}


def _save_cache(pages: dict[int, str]) -> None:
    OCR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ordered = [{"page": p, "text": pages[p]} for p in sorted(pages)]
    OCR_PATH.write_text(
        json.dumps({"document": DOCUMENT, "pages": ordered}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    reader = PdfReader(str(DOCS_DIR / SOURCE_PDF))
    pages = _load_cache()
    # Re-OCR pages that are missing or cached empty (a transient empty response).
    todo = [p for p in FOCUS_PAGES if not pages.get(p, "").strip()]
    if not todo:
        print(f"All {len(FOCUS_PAGES)} focus pages already cached at {OCR_PATH}.")
        return

    print(f"OCR-ing {len(todo)} of {len(FOCUS_PAGES)} focus pages with Gemini vision ...")
    for i, printed_page in enumerate(todo):
        image_bytes = _page_image_jpeg(reader, FOCUS_PAGES.index(printed_page))
        text = _ocr_with_retry(image_bytes, printed_page)
        pages[printed_page] = text
        _save_cache(pages)  # persist after each page so progress is never lost
        print(f"  page {printed_page}: {len(text)} chars")
        if i < len(todo) - 1:
            time.sleep(DELAY_SECONDS)

    print(f"Done. Cached {len(pages)} pages to {OCR_PATH}.")


if __name__ == "__main__":
    main()
