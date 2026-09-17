# core/ai_extract.py
"""
Free AI-based document text + STRUCTURED FIELD extraction using Google
Gemini's free API tier.

This replaces the old Tesseract + OpenCV OCR pipeline in core/ocr_utils.py
(process_document_file_enhanced), which needed a system-level Tesseract
binary installed on the server. Tesseract is disabled for now — this module
sends the document straight to Gemini (which reads images and PDFs
natively, no separate OCR step).

Two extraction modes, chosen automatically based on `doc_type`:

  1. STRUCTURED (doc_type is one of models.DOCUMENT_FIELD_TEMPLATES): asks
     Gemini to return JSON matching that document type's *exact predefined
     field names* directly (responseSchema-constrained), with an explicit
     instruction to leave a field empty rather than guess. This is the
     primary path — it's what lets the chatbot return the exact stored
     value for a field with no separate regex-guessing step.
  2. GENERIC (doc_type unknown/"other_document"/not provided): falls back
     to plain OCR transcription. Downstream code (core/utils.py) then runs
     the legacy regex-based extractor over the raw text as a best-effort
     fallback, same as before.

If the structured call fails for any reason (bad JSON, blocked prompt,
network error), we transparently retry with the generic OCR-only prompt
rather than failing the whole upload — never guess field values, but do
still get the text.

Reuses the SAME GEMINI_API_KEY already configured for core/ai_chat.py — no
extra signup or key needed. Get a free key (no credit card required) at:
https://aistudio.google.com/apikey

Output shape (backward compatible with the old raw-text-only version —
callers that only look at `raw_text` keep working unchanged):

    {
      "page_1": {
        "raw_text": "...",          # full transcription, always present
        "fields": {...} or absent,  # predefined key/value pairs (structured mode only)
      },
      "page_2": {...},              # generic mode may have multiple pages
      "_summary": {...},
    }

or, on failure:

    {"error": "human readable reason"}
"""
import base64
import json as json_lib
import logging
import mimetypes
import os
from typing import Dict, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-flash-latest"  # Google's rolling alias for "whatever
# the current Flash model is" - hot-swapped by Google on every new release,
# so this keeps working across model retirements instead of pointing at a
# dated model name (e.g. gemini-1.5-flash) that eventually gets shut down
# and starts returning 404 on every request.
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

GENERIC_EXTRACTION_PROMPT = (
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


def _read_file_b64(file_path: str):
    if not os.path.exists(file_path):
        return None, {"error": "File not found on disk."}
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        return None, {"error": "File is too large for AI extraction (max 15MB)."}
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8"), None
    except Exception as e:
        logger.error("AI extraction: failed to read file from disk")
        return None, {"error": f"Could not read file: {e}"}


def _call_gemini(api_key: str, parts: list, generation_config: dict) -> Dict:
    """Low-level Gemini call. Returns {'candidates': ...} or {'error': ...}."""
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config,
    }
    try:
        response = requests.post(
            GEMINI_URL, params={"key": api_key}, json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"AI extraction request failed: {e}")
        return {"error": f"AI extraction service unavailable right now: {e}"}
    except ValueError:
        logger.error("AI extraction returned invalid JSON")
        return {"error": "AI extraction service returned an invalid response."}


def _extract_text_from_response(data: Dict):
    candidates = data.get("candidates", [])
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no text detected")
        return None, f"AI extraction found nothing usable ({reason})."
    try:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        logger.error("Unexpected AI extraction response shape")
        return None, "AI extraction returned an unexpected response format."
    if not text:
        return None, "No text could be extracted from this document."
    return text, None


def _build_field_schema(field_names) -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "raw_text": {"type": "STRING"},
            "fields": {
                "type": "OBJECT",
                "properties": {name: {"type": "STRING"} for name in field_names},
            },
        },
        "required": ["raw_text", "fields"],
    }


def _build_structured_prompt(doc_type_label: str, field_names) -> str:
    field_list = "\n".join(f'  - "{name}"' for name in field_names)
    return (
        f"You are reading a scanned {doc_type_label} (image or PDF). "
        "First, transcribe the full visible text into `raw_text` exactly as printed. "
        "Then fill in `fields` using EXACTLY these predefined keys - do not rename, "
        "translate, reformat, or add extra keys:\n"
        f"{field_list}\n\n"
        "Rules:\n"
        "- Copy each value exactly as it appears on the document (same digits, "
        "same date format, same spelling).\n"
        "- If a field is not visible on the document, set it to an empty string \"\" - "
        "NEVER guess, infer, or invent a value for a missing field.\n"
        "- Return nothing except the JSON object described by the schema."
    )


def _try_structured_extraction(api_key: str, encoded: str, mime_type: str, doc_type_label: str, field_names) -> Optional[Dict]:
    """Returns a result dict on success, or None to signal 'fall back to generic'."""
    prompt = _build_structured_prompt(doc_type_label, field_names)
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": encoded}},
    ]
    generation_config = {
        "temperature": 0.0,
        "maxOutputTokens": 4096,
        "responseMimeType": "application/json",
        "responseSchema": _build_field_schema(field_names),
    }
    data = _call_gemini(api_key, parts, generation_config)
    if "error" in data:
        logger.warning("Structured extraction call failed, falling back to generic OCR")
        return None

    text, err = _extract_text_from_response(data)
    if err:
        logger.warning(f"Structured extraction produced no usable text ({err}); falling back")
        return None

    try:
        parsed = json_lib.loads(text)
    except (json_lib.JSONDecodeError, TypeError):
        logger.warning("Structured extraction returned non-JSON text; falling back to generic OCR")
        return None

    raw_text = (parsed.get("raw_text") or "").strip()
    fields = parsed.get("fields") or {}
    if not isinstance(fields, dict):
        fields = {}
    # Drop empty-string fields the model correctly declined to guess, and
    # anything that isn't one of the predefined keys we asked for.
    clean_fields = {
        k: v.strip() if isinstance(v, str) else v
        for k, v in fields.items()
        if k in field_names and v not in (None, "", [])
    }

    if not raw_text and not clean_fields:
        return None

    return {
        "page_1": {"raw_text": raw_text, "fields": clean_fields},
        "_summary": {
            "total_pages": 1,
            "successful_pages": 1,
            "failed_pages": 0,
            "extraction_method": "gemini_structured",
        },
    }


def _generic_extraction(api_key: str, encoded: str, mime_type: str) -> Dict:
    parts = [
        {"text": GENERIC_EXTRACTION_PROMPT},
        {"inline_data": {"mime_type": mime_type, "data": encoded}},
    ]
    generation_config = {"temperature": 0.0, "maxOutputTokens": 4096}
    data = _call_gemini(api_key, parts, generation_config)
    if "error" in data:
        return data

    text, err = _extract_text_from_response(data)
    if err:
        return {"error": err}

    pages = [p.strip() for p in text.split(PAGE_SEPARATOR) if p.strip()] or [text]
    result: Dict = {}
    for idx, page_text in enumerate(pages, start=1):
        result[f"page_{idx}"] = {"raw_text": page_text}
    result["_summary"] = {
        "total_pages": len(pages),
        "successful_pages": len(pages),
        "failed_pages": 0,
        "extraction_method": "gemini_generic",
    }
    return result


def extract_document_ai(
    file_path: str,
    doc_type: Optional[str] = None,
    auto_detect: bool = True,
) -> Dict:
    """
    Extract text (and, when doc_type is a known type, exact predefined
    structured fields) from a document using Gemini's free tier.

    `auto_detect` is accepted for drop-in compatibility with the old
    process_document_file_enhanced() signature; document-type auto-detection
    for unknown/generic uploads still happens downstream in
    ai_utils.detect_document_type() once raw text comes back.
    """
    api_key = getattr(settings, "GEMINI_API_KEY", "")
    if not api_key:
        return {
            "error": (
                "AI extraction isn't configured (missing GEMINI_API_KEY). "
                "Get a free key at https://aistudio.google.com/apikey and add it to your .env"
            )
        }

    encoded, err = _read_file_b64(file_path)
    if err:
        return err

    mime_type = _guess_mime_type(file_path)

    # Local import avoids any import-order issues with the Django app
    # registry (models.py must be fully loaded before we can read it).
    field_names = None
    doc_type_label = None
    if doc_type:
        from .models import DOCUMENT_FIELD_TEMPLATES, Document

        template = DOCUMENT_FIELD_TEMPLATES.get(doc_type)
        if template:
            field_names = list(template.keys())
            doc_type_label = dict(Document.DOC_TYPES).get(doc_type, doc_type.replace("_", " ").title())

    if field_names:
        structured_result = _try_structured_extraction(api_key, encoded, mime_type, doc_type_label, field_names)
        if structured_result is not None:
            return structured_result
        # Fall through to generic extraction below - never hard-fail an
        # upload just because structured/schema mode didn't pan out.

    return _generic_extraction(api_key, encoded, mime_type)


def ai_extraction_available() -> bool:
    """Whether AI extraction is currently usable (i.e. a key is configured)."""
    return bool(getattr(settings, "GEMINI_API_KEY", ""))
