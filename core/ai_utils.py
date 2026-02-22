import json
import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# =====================================================
# 🚫 OpenAI Disabled (Free Mode)
# =====================================================

client = None
logger.info("Running in FREE rule-based extraction mode. OpenAI disabled.")


# =====================================================
# 🧹 Clean JSON safely
# =====================================================
def clean_json_response(content: str) -> Dict:
    """
    Safely parse JSON string if needed.
    """
    if not content:
        return {}

    try:
        content = re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(content[start : end + 1])
    except Exception:
        return {}


# =====================================================
# 🔐 Mask sensitive data (Optional utility)
# =====================================================
def mask_sensitive_data(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX XXXX XXXX", text)
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "XXXXX0000X", text)

    return text


# =====================================================
# 🧼 Clean OCR text
# =====================================================
def clean_ocr_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# =====================================================
# 📄 Document Type Detection
# =====================================================
def detect_document_type(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    cleaned = clean_ocr_text(text).lower()

    if len(cleaned) < 20:
        return None

    if "aadhaar" in cleaned or "uidai" in cleaned:
        return "Aadhaar Card"

    if "income tax department" in cleaned and "permanent account number" in cleaned:
        return "PAN Card"

    if "driving licence" in cleaned or "transport department" in cleaned:
        return "Driving License"

    if "passport" in cleaned and "republic of india" in cleaned:
        return "Passport"

    if "invoice" in cleaned or "gst" in cleaned:
        return "Invoice"

    return "Other_Document"


# =====================================================
# 🤖 FREE Multi-Document Structured Extraction Engine
# =====================================================
def extract_structured_data(text: str) -> Dict:

    if not text or not text.strip():
        return {}

    text = clean_ocr_text(text)
    lowered = text.lower()

    # -------------------------------------------------
    # Document Routing
    # -------------------------------------------------
    if "aadhaar" in lowered or "uidai" in lowered:
        return extract_aadhaar(text)

    if "income tax department" in lowered and "permanent account number" in lowered:
        return extract_pan(text)

    if "driving licence" in lowered or "transport department" in lowered:
        return extract_driving_license(text)

    if "passport" in lowered and "republic of india" in lowered:
        return extract_passport(text)

    if "invoice" in lowered or "gst" in lowered:
        return extract_invoice(text)

    return generic_extraction(text)


# =====================================================
# 🟢 AADHAAR CARD
# =====================================================
def extract_aadhaar(text: str) -> Dict:
    data = {"detected_document_type": "Aadhaar Card"}

    # Aadhaar number (12 digits with or without spaces)
    aadhaar = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
    if aadhaar:
        data["aadhaar_number"] = aadhaar.group()

    # Name (usually uppercase line)
    name = re.search(r"\n([A-Z][A-Za-z ]{3,})\n", text)
    if name:
        data["name"] = name.group(1).strip()

    # DOB (common formats)
    dob = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if dob:
        data["date_of_birth"] = dob.group()

    # Gender
    gender = re.search(r"\b(MALE|FEMALE|Male|Female)\b", text)
    if gender:
        data["gender"] = gender.group()

    return data


# =====================================================
# 🟢 PAN CARD (Income Tax Department)
# =====================================================
def extract_pan(text: str) -> Dict:
    data = {"detected_document_type": "PAN Card"}

    pan = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    if pan:
        data["pan_number"] = pan.group()

    name_match = re.search(r"\n([A-Z][A-Za-z ]+)\n", text)
    if name_match:
        data["name"] = name_match.group(1).strip()

    dob = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if dob:
        data["date_of_birth"] = dob.group()

    return data


# =====================================================
# 🟢 DRIVING LICENSE
# =====================================================
def extract_driving_license(text: str) -> Dict:
    data = {"detected_document_type": "Driving License"}

    dl_number = re.search(r"\b[A-Z]{2}\d{2}\s?\d{11}\b", text)
    if dl_number:
        data["license_number"] = dl_number.group()

    dob = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if dob:
        data["date_of_birth"] = dob.group()

    name = re.search(r"\n([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", text)
    if name:
        data["name"] = name.group(1)

    address_match = re.search(r"address[:\s]*(.*)", text, re.IGNORECASE)
    if address_match:
        data["address"] = address_match.group(1).strip()

    return data


# =====================================================
# 🟢 PASSPORT
# =====================================================
def extract_passport(text: str) -> Dict:
    data = {"detected_document_type": "Passport"}

    passport_no = re.search(r"\b[A-Z][0-9]{7}\b", text)
    if passport_no:
        data["passport_number"] = passport_no.group()

    dob = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if dob:
        data["date_of_birth"] = dob.group()

    nationality = re.search(r"nationality\s*[:\-]?\s*([A-Za-z ]+)", text, re.IGNORECASE)
    if nationality:
        data["nationality"] = nationality.group(1).strip()

    name = re.search(r"\n([A-Z][A-Za-z ]+)\n", text)
    if name:
        data["name"] = name.group(1).strip()

    return data


# =====================================================
# 🟢 INVOICE
# =====================================================
def extract_invoice(text: str) -> Dict:
    data = {"detected_document_type": "Invoice"}

    invoice_no = re.search(
        r"invoice\s*(no|number)?[:\s]*([A-Za-z0-9\-\/]+)", text, re.IGNORECASE
    )
    if invoice_no:
        data["invoice_number"] = invoice_no.group(2)

    gst = re.search(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ\d\b", text)
    if gst:
        data["gst_number"] = gst.group()

    total = re.search(
        r"(total\s*amount|grand total)[:\s₹]*([\d,]+\.\d{2})", text, re.IGNORECASE
    )
    if total:
        data["total_amount"] = total.group(2)

    date = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if date:
        data["invoice_date"] = date.group()

    return data


# Generic fallback extraction
# ----------------------------------------
def generic_extraction(text: str) -> Dict:
    data = {}

    email = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    if email:
        data["email"] = email.group()

    phone = re.search(r"\b[6-9]\d{9}\b", text)
    if phone:
        data["phone"] = phone.group()

    return data
