# C:\chatbot\ask_me\core\ai_utils.py

from openai import OpenAI
import os
import json
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# =========================================
# 🔐 Safe OpenAI client initialization
# =========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("OPENAI KEY FROM AI_UTILS:", os.getenv("OPENAI_API_KEY"))

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"OpenAI initialization failed: {e}")
else:
    logger.warning("OPENAI_API_KEY not found. AI extraction disabled.")


# =========================================
# 🧹 Clean JSON safely from model response
# =========================================
def clean_json_response(content: str) -> Dict:
    """
    Extract and safely parse JSON from model output.
    Handles markdown, extra text, malformed wrapping.
    """

    if not content:
        return {}

    try:
        # remove markdown fences
        content = re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()

        # extract JSON object boundaries
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1:
            return {}

        json_str = content[start : end + 1]

        return json.loads(json_str)

    except Exception as e:
        logger.warning(f"JSON cleaning failed: {e}")
        return {}


# =========================================
# 🔐 Mask sensitive data before AI call
# =========================================
def mask_sensitive_data(text: str) -> str:
    """
    Mask highly sensitive identifiers before sending to AI.
    """

    if not text:
        return ""

    # Aadhaar number (12 digits)
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX XXXX XXXX", text)

    # PAN number
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXX0000X", text)

    return text


# =========================================
# 🧼 Clean OCR text before any AI processing
# =========================================
def clean_ocr_text(text: str) -> str:
    """
    Clean OCR-extracted text to improve AI parsing.
    - Remove excessive whitespace
    - Normalize line breaks
    - Remove stray non-printable characters
    """
    if not text:
        return ""

    # Remove control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n\s*\n", "\n", text)

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


# =========================================
# 📄 Document Type Detection (OCR-based)
# =========================================
def detect_document_type(text: str) -> Optional[str]:
    """
    Detect document type from OCR extracted text.
    Lightweight rule-based detection.
    """

    if not text or not isinstance(text, str):
        logger.warning("Document detection skipped: invalid OCR text")
        return None

    # Clean OCR text for better rule matching (optional but beneficial)
    cleaned = clean_ocr_text(text)
    cleaned = cleaned.lower()

    if len(cleaned) < 20:
        logger.warning("Document detection skipped: text too short")
        return None

    if "aadhaar" in cleaned or "uidai" in cleaned:
        return "Aadhaar Card"

    if "income tax department" in cleaned and "permanent account number" in cleaned:
        return "PAN Card"

    if "driving licence" in cleaned or "transport department" in cleaned:
        return "Driving License"

    if "passport" in cleaned and "republic of india" in cleaned:
        return "Passport"

    return "Other_Document"


# =========================================
# 🤖 AI Structured Data Extraction
# =========================================
def extract_structured_data(text: str) -> Dict:
    """
    Extract structured JSON from document text using OpenAI.
    Returns dictionary or {}.
    """

    global client

    # -----------------------------------------
    # Safety checks
    # -----------------------------------------
    if not client:
        logger.warning("AI extraction skipped: OpenAI client not configured")
        return {}

    if not text or not text.strip():
        logger.warning("AI extraction skipped: empty text")
        return {}

    # -----------------------------------------
    # Step 1 — Clean OCR text
    # -----------------------------------------
    cleaned_text = clean_ocr_text(text)

    # -----------------------------------------
    # Step 2 — Mask sensitive data
    # -----------------------------------------
    safe_text = mask_sensitive_data(cleaned_text)

    try:
        # -----------------------------------------
        # Step 3 — OpenAI extraction
        # -----------------------------------------
        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=0,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract structured key-value JSON from the document text.\n"
                        "Return ONLY valid JSON.\n"
                        "No explanation.\n"
                        "No markdown.\n"
                        "Do NOT guess missing values.\n"
                        "If nothing found return {}."
                    ),
                },
                {
                    "role": "user",
                    "content": safe_text,
                },
            ],
        )

        content = response.output_text or ""

        # -----------------------------------------
        # Step 4 — Clean JSON
        # -----------------------------------------
        result = clean_json_response(content)

        # -----------------------------------------
        # Step 5 — Validate result
        # -----------------------------------------
        if not isinstance(result, dict):
            logger.warning("AI returned non-dictionary value")
            return {}

        if not result:
            logger.warning("AI returned empty JSON")

        return result

    except Exception as e:
        logger.error(f"AI extraction failed: {e}")

        # -----------------------------------------
        # Disable AI if API key invalid
        # -----------------------------------------
        error_text = str(e).lower()
        if "401" in error_text or "invalid_api_key" in error_text:
            logger.critical("Invalid OpenAI API key detected. Disabling AI extraction.")
            client = None

        return {}
