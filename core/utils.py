# core/utils.py - Shared utilities
import json
import logging
from typing import Dict, Any, Optional

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
