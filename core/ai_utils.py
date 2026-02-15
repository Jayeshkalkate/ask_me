# C:\chatbot\ask_me\core\ai_utils.py

from openai import OpenAI
import os
import json
import re


# 🔐 Safe OpenAI client initialization
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# =========================================
# Clean JSON safely from model response
# =========================================
def clean_json_response(content: str):
    """
    Extract valid JSON from model response.
    Handles ```json ``` wrapping and invalid formatting.
    Returns dict or {}.
    """
    try:
        # Remove markdown code block if present
        content = re.sub(r"```json|```", "", content).strip()

        # Remove leading text before JSON starts
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
    """
    Mask sensitive information before sending to AI.
    """

    # Aadhaar (12 digits)
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX XXXX XXXX", text)

    # PAN
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXX0000X", text)

    return text


# =========================================
# MAIN AI EXTRACTION FUNCTION
# =========================================
def extract_structured_data(text):
    """
    Extract structured JSON from document text using AI.

    RULES:
    ✔ Return ONLY JSON dict
    ✔ No fallback template
    ✔ If failure → return {}
    ✔ If OpenAI not configured → {}
    ✔ If model returns invalid JSON → {}
    """

    # 🚨 OpenAI not configured
    if not client:
        print("OpenAI not configured")
        return {}

    # 🔐 Privacy masking
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

        # Final safety check
        if isinstance(cleaned_json, dict):
            return cleaned_json

        return {}

    except Exception as e:
        print("AI Error:", e)
        return {}
