# C:\chatbot\ask_me\core\ai_utils.py

from openai import OpenAI
import os
import json
import re

# 🔐 Safe OpenAI client initialization
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


def clean_json_response(content: str):
    """
    Safely extract JSON from AI response.
    Handles cases where model wraps JSON in ```json ``` blocks.
    """
    try:
        content = re.sub(r"```json|```", "", content).strip()
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def mask_sensitive_data(text: str) -> str:
    """
    Mask sensitive information before sending to AI.
    """
    # Mask Aadhaar (12 digit format)
    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX XXXX XXXX", text)

    # Mask PAN (ABCDE1234F format)
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXX0000X", text)

    return text


def extract_document_data(text, doc_type=None):
    """
    AI-powered structured extraction.
    Returns clean JSON matching your templates.
    """

    # 🔹 Define dynamic templates
    if doc_type == "aadhaar_card":
        template = {
            "Full Name": "",
            "Aadhaar Number": "",
            "Date of Birth": "",
            "Gender": "",
            "Address": "",
            "confidence_score": 0,
        }

    elif doc_type == "pan_card":
        template = {
            "Full Name": "",
            "PAN Number": "",
            "Date of Birth": "",
            "Father's Name": "",
            "confidence_score": 0,
        }

    elif doc_type == "driving_license":
        template = {
            "Full Name": "",
            "License Number": "",
            "Date of Birth": "",
            "Address": "",
            "Valid Until": "",
            "confidence_score": 0,
        }

    else:
        template = {
            "Document Type": "",
            "Content": "",
            "confidence_score": 0,
        }

    # 🚨 If no API key available → skip AI safely
    if not client:
        return template

    # 🔐 Mask sensitive data before sending to OpenAI
    safe_text = mask_sensitive_data(text)

    # 🔐 Strict extraction prompt
    prompt = f"""
You are a secure document extraction AI.

Strict Rules:
- Extract ONLY data present in the text.
- Do NOT guess or hallucinate.
- If field not found, leave it empty.
- Return ONLY valid JSON.
- Include confidence_score from 0-100 based on extraction certainty.

Return JSON in this exact structure:
{json.dumps(template, indent=2)}

Document Text:
{safe_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content
        parsed_json = clean_json_response(content)

        # Ensure template structure consistency
        final_data = template.copy()
        final_data.update(parsed_json)

        return final_data

    except Exception:
        return template
