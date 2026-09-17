# core/utils.py - Shared utilities
import json
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Internal keys to skip in extraction
INTERNAL_KEYS = {"_metadata", "status", "raw_text", "structured_data"}


def clean_extracted_data(data: Dict) -> Dict:
    """
    Recursively remove internal keys from extracted data.
    Returns a new dict with only meaningful fields.
    """
    if not isinstance(data, dict):
        return data

    cleaned = {}
    for key, value in data.items():
        if key in INTERNAL_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = clean_extracted_data(value)
        else:
            cleaned[key] = value
    return cleaned


def build_document_text_and_fields(ocr_result: Dict) -> "tuple[str, Dict]":
    """
    Turn the dict returned by ai_extract.extract_document_ai() into
    (ocr_text, final_data) ready to save on Document.extracted_text /
    Document.extracted_data.

    Prefers the AI's own structured `fields` (extracted directly against the
    document type's predefined schema - see ai_extract.py's structured
    mode) over the legacy regex-based extractor in ai_utils.py, which now
    only runs as a fallback when no schema-based fields came back (unknown
    doc type, or the model couldn't find anything on the page).
    """
    ocr_parts = []
    merged_fields: Dict[str, Any] = {}

    for page_key, page_data in (ocr_result or {}).items():
        if page_key.startswith("_") or not isinstance(page_data, dict):
            continue
        raw_text = page_data.get("raw_text")
        if raw_text:
            ocr_parts.append(raw_text)
        page_fields = page_data.get("fields") or {}
        if isinstance(page_fields, dict):
            for key, value in page_fields.items():
                if value and key not in merged_fields:
                    merged_fields[key] = value

    ocr_text = " ".join(ocr_parts).strip()

    if merged_fields:
        final_data = {"page_1": merged_fields}
    elif ocr_text:
        structured_data = get_structured_fields_from_text(ocr_text)
        if structured_data:
            final_data = {"page_1": structured_data}
        else:
            final_data = {"page_1": {"Content": ocr_text[:500]}}
    else:
        final_data = {"page_1": {}}

    return ocr_text, final_data


def get_structured_fields_from_text(ocr_text: str) -> Dict:
    """
    Use rule‑based extraction to get structured fields.
    Returns a dict of fields, or empty dict if none.
    """
    if not ocr_text or len(ocr_text.strip()) < 30:
        return {}

    # Try the advanced extractor
    from .ai_utils import extract_structured_data, generic_extraction
    structured = extract_structured_data(ocr_text)
    if structured and isinstance(structured, dict):
        # Remove any internal keys that might have slipped in
        return clean_extracted_data(structured)

    # Fallback: generic extraction
    try:
        fallback = generic_extraction(ocr_text)
        if fallback:
            return fallback
    except Exception as e:
        logger.warning(f"Fallback extraction failed: {e}")

    return {}


def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    return obj