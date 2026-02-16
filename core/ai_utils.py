# C:\chatbot\ask_me\core\ai_utils.py

from openai import OpenAI
import os
import json
import re


# =========================================
# 🔐 Safe OpenAI client initialization
# =========================================
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# =========================================
# Clean JSON safely from model response
# =========================================
def clean_json_response(content: str):
    try:
        content = re.sub(r"```json|```", "", content).strip()

        start = content.find("{")
        end = content.rfind("}")

        if start != -1 and end != -1:
            content = content[start : end + 1]

        return json.loads(content)

    except Exception:
        return {}


# =========================================
# Mask sensitive data before AI call
# =========================================
def mask_sensitive_data(text: str) -> str:

    # Aadhaar
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX XXXX XXXX", text)

    # PAN
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXX0000X", text)

    return text


# =========================================
# 🔥 DOCUMENT TYPE DETECTION (FIXED)
# =========================================
def detect_document_type(text):
    """
    Detect document type based on OCR extracted text.
    """

    if not text or not isinstance(text, str):
        print("⚠️ Detection skipped: Empty or invalid OCR text")
        return None

    text = text.strip()

    if len(text) < 20:
        print("⚠️ Detection skipped: OCR text too short")
        return None

    text_lower = text.lower()

    if "government of india" in text_lower and "aadhaar" in text_lower:
        return "Aadhaar Card"

    if (
        "income tax department" in text_lower
        and "permanent account number" in text_lower
    ):
        return "PAN Card"

    if "driving licence" in text_lower or "transport department" in text_lower:
        return "Driving License"

    if "passport" in text_lower and "republic of india" in text_lower:
        return "Passport"

    return "Other_Document"


# =========================================
# 🔥 MAIN AI EXTRACTION
# =========================================
def extract_structured_data(text):

    if not client:
        print("⚠️ OpenAI not configured")
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
                        "No guessing missing values.\n"
                        "If nothing found return {}."
                    ),
                },
                {
                    "role": "user",
                    "content": safe_text,
                },
            ],
        )

        content = response.choices[0].message.content
        cleaned_json = clean_json_response(content)

        if isinstance(cleaned_json, dict):
            return cleaned_json

        return {}

    except Exception as e:
        print("AI Error:", e)
        return {}
