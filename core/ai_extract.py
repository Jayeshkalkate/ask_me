# core/ai_extract.py
"""
Free AI-based document text extraction using Google Gemini's free API tier.

This replaces the old Tesseract + OpenCV OCR pipeline in core/ocr_utils.py
(process_document_file_enhanced), which needed a system-level Tesseract
binary installed on the server. Tesseract is disabled for now — this module
sends the document straight to Gemini (which reads images and PDFs
natively, no separate OCR step) and gets the raw text back.

It reuses the SAME GEMINI_API_KEY already configured for core/ai_chat.py
(the chatbot feature) — no extra signup or key needed. Get a free key
(no credit card required) at: https://aistudio.google.com/apikey

Everything downstream (document-type detection, key/value field
extraction, cleaning, storage, offline caching/search) is unchanged: it
already works purely off the returned raw text, in ai_utils.py / utils.py.

Output shape matches what process_document_file_enhanced() used to return,
so callers (tasks.py, views.py, views_offline.py) don't need to change how
they read the result:

    {"page_1": {"raw_text": "..."}, "page_2": {...}, "_summary": {...}}

or, on failure:

    {"error": "human readable reason"}

There is intentionally NO Tesseract fallback right now. If GEMINI_API_KEY
isn't set or the request fails, extraction fails outright and the document
is marked unprocessed/errored so it can be retried later (see reprocess
document in views.py) once AI extraction is available again.
"""
import base64
import logging
import mimetypes
import os
from typing import Dict, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
REQUEST_TIMEOUT = 45  # extraction takes longer than the short chat replies in ai_chat.py
MAX_FILE_SIZE = 15 * 1024 * 1024  # stay comfortably inside Gemini's inline-data limit

_MIME_OVERRIDES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".pdf": "application/pdf",
}

PAGE_SEPARATOR = "---PAGE---"

EXTRACTION_PROMPT = (
    "You are an OCR engine. Read every page of this document (image or PDF) and "
    "return ONLY the raw text you can see, preserving line breaks and the original "
    "layout as closely as possible (e.g. 'Name: Jay Ram Mali' on its own line). "
    "Do not summarize, translate, or add any commentary - just transcribe the text "
    "exactly as printed. If the document has more than one page, separate each "
    f"page's transcribed text with a line containing exactly: {PAGE_SEPARATOR}"
)


def _guess_mime_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[ext]
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def extract_document_ai(
    file_path: str,
    doc_type: Optional[str] = None,
    auto_detect: bool = True,
) -> Dict:
    """
    Extract text from a document (image or PDF) using Gemini's free tier.

    `doc_type` / `auto_detect` are accepted for drop-in compatibility with
    the old process_document_file_enhanced() signature but aren't used here
    directly - document-type auto-detection already happens downstream
    (see ai_utils.detect_document_type) once the raw text comes back.
    """
    api_key = getattr(settings, "GEMINI_API_KEY", "")
    if not api_key:
        return {
            "error": (
                "AI extraction isn't configured (missing GEMINI_API_KEY). "
                "Get a free key at https://aistudio.google.com/apikey and add it to your .env"
            )
        }

    if not os.path.exists(file_path):
        return {"error": "File not found on disk."}

    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        return {"error": "File is too large for AI extraction (max 15MB)."}

    mime_type = _guess_mime_type(file_path)

    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"AI extraction: failed to read file: {e}")
        return {"error": f"Could not read file: {e}"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
    }

    try:
        response = requests.post(
            GEMINI_URL, params={"key": api_key}, json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"AI extraction request failed: {e}")
        return {"error": f"AI extraction service unavailable right now: {e}"}
    except ValueError as e:
        logger.error(f"AI extraction returned invalid JSON: {e}")
        return {"error": "AI extraction service returned an invalid response."}

    try:
        candidates = data.get("candidates", [])
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no text detected")
            return {"error": f"AI extraction found nothing usable ({reason})."}
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Unexpected AI extraction response shape: {e}")
        return {"error": "AI extraction returned an unexpected response format."}

    if not text:
        return {"error": "No text could be extracted from this document."}

    pages = [p.strip() for p in text.split(PAGE_SEPARATOR) if p.strip()] or [text]

    result: Dict = {}
    for idx, page_text in enumerate(pages, start=1):
        result[f"page_{idx}"] = {"raw_text": page_text}

    result["_summary"] = {
        "total_pages": len(pages),
        "successful_pages": len(pages),
        "failed_pages": 0,
        "extraction_method": "gemini_ai",
    }
    return result


def ai_extraction_available() -> bool:
    """Whether AI extraction is currently usable (i.e. a key is configured)."""
    return bool(getattr(settings, "GEMINI_API_KEY", ""))
