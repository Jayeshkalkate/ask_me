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

    text = text.strip().lower()

    if len(text) < 20:
        logger.warning("Document detection skipped: text too short")
        return None

    if "government of india" in text and "aadhaar" in text:
        return "Aadhaar Card"

    if "income tax department" in text and "permanent account number" in text:
        return "PAN Card"

    if "driving licence" in text or "transport department" in text:
        return "Driving License"

    if "passport" in text and "republic of india" in text:
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

    if not client:
        logger.warning("AI extraction skipped: OpenAI client not configured")
        return {}

    if not text or not text.strip():
        logger.warning("AI extraction skipped: empty text")
        return {}

    safe_text = mask_sensitive_data(text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
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

        content = response.choices[0].message.content or ""
        result = clean_json_response(content)

        return result if isinstance(result, dict) else {}

    except Exception as e:
        logger.error(f"AI extraction failed: {e}")
        return {}
