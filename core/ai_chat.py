# core/ai_chat.py
"""
Free-tier AI chat layer for ASK_ME.

Uses Google Gemini's free API tier (no billing required) to generate real,
conversational answers grounded ONLY in the user's own document data that the
existing rapidfuzz-based matching in views.py already retrieved.

If GEMINI_API_KEY isn't set, the API call fails, or the response can't be
parsed, every function here returns None - callers must fall back to the
existing rule-based response logic so the chat feature never breaks even
without an API key or if the free quota runs out.

Get a free key (no credit card needed) at: https://aistudio.google.com/apikey
"""
import json
import logging
from typing import Dict, List, Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
REQUEST_TIMEOUT = 12  # seconds - keep short so chat_api doesn't hang if Gemini is slow/down
MAX_CONTEXT_DOCS = 3
MAX_TEXT_SNIPPET_CHARS = 500


def _build_context_text(context_docs: List[Dict]) -> str:
    """Turn a list of {'doc_type': ..., 'fields': {...}, 'extracted_text': ...} into plain text."""
    parts = []
    for i, doc in enumerate(context_docs[:MAX_CONTEXT_DOCS], start=1):
        parts.append(f"Document {i} ({doc.get('doc_type', 'unknown type')}):")
        for field, value in doc.get("fields", {}).items():
            parts.append(f"  - {field}: {value}")
        snippet = (doc.get("extracted_text") or "")[:MAX_TEXT_SNIPPET_CHARS]
        if snippet:
            parts.append(f"  Extracted text snippet: {snippet}")
    return "\n".join(parts)


def generate_ai_answer(user_message: str, context_docs: List[Dict]) -> Optional[str]:
    """
    Ask Gemini to answer `user_message` using ONLY the given document context.

    Returns the answer text, or None if:
      - GEMINI_API_KEY isn't configured
      - there's no context to ground the answer in
      - the HTTP request fails or times out
      - the response can't be parsed

    Callers (views.py::_handle_chat_query) must fall back to the existing
    rapidfuzz-based response when this returns None.
    """
    api_key = getattr(settings, "GEMINI_API_KEY", "")
    if not api_key:
        return None
    if not context_docs:
        return None

    context_text = _build_context_text(context_docs)
    prompt = (
        "You are a helpful assistant answering questions about a user's own uploaded "
        "documents. Only use the information given below - never guess or invent values. "
        "If the answer isn't in the provided data, say plainly that you couldn't find it "
        "in their documents. Keep the answer short (1-3 sentences) and direct.\n\n"
        f"Document data:\n{context_text}\n\n"
        f"Question: {user_message}\n"
        "Answer:"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 300,
        },
    }

    try:
        response = requests.post(
            GEMINI_URL,
            params={"key": api_key},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"Gemini API request failed, falling back to rule-based chat: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"Gemini API returned invalid JSON: {e}")
        return None

    try:
        candidates = data.get("candidates", [])
        if not candidates:
            # Common cause: prompt was blocked by Gemini's safety filters.
            logger.info(f"Gemini returned no candidates: {data.get('promptFeedback')}")
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return text or None
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Unexpected Gemini response shape: {e}")
        return None


def build_context_from_matches(best_matches: List[tuple]) -> List[Dict]:
    """
    Convert the (doc, score, ...) tuples already produced by the existing
    rapidfuzz matching in views.py::_handle_chat_query into the context format
    generate_ai_answer() expects - deduplicated by document, best matches first.
    """
    from .utils import INTERNAL_KEYS  # local import avoids a circular import with views.py

    seen_doc_ids = set()
    context_docs = []

    for match in best_matches:
        doc = match[0]
        if doc.id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc.id)

        fields = {}
        display_data = doc.display_data or {}
        for page_data in display_data.values():
            if not isinstance(page_data, dict):
                continue
            for field_key, field_value in page_data.items():
                if field_key.startswith("_") or field_key in INTERNAL_KEYS or not field_value:
                    continue
                fields[field_key] = field_value

        context_docs.append({
            "doc_type": doc.get_doc_type_display(),
            "fields": fields,
            "extracted_text": doc.extracted_text or "",
        })

        if len(context_docs) >= MAX_CONTEXT_DOCS:
            break

    return context_docs
