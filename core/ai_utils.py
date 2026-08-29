import json
import re
import logging
from typing import Dict, Optional, Any, List, Pattern, Tuple

logger = logging.getLogger(__name__)

# ============================================================
#  CONFIGURATION
# ============================================================
MIN_TEXT_LENGTH_FOR_DETECTION = 20
MIN_TEXT_LENGTH_FOR_EXTRACTION = 30

# ============================================================
#  COMPILED REGEX PATTERNS
# ============================================================

# --- Identity Documents ---
# Aadhaar: 12 digits with optional spaces/hyphens
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
# PAN: 5 uppercase letters, 4 digits, 1 uppercase letter
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
# Voter ID: 3 letters + 7 digits (EPIC format)
VOTER_ID_PATTERN = re.compile(r"\b[A-Z]{3}[0-9]{7}\b")
# Passport: 1 uppercase letter + 7 digits
PASSPORT_PATTERN = re.compile(r"\b[A-Z][0-9]{7}\b")
# Driving License: state code + 2 digits + 11 digits (common format)
DL_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}\s?\d{11}\b")
# Vehicle Registration: state code + 2 digits + 4 letters + 4 digits
VEHICLE_REG_PATTERN = re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z]{4}[0-9]{4}\b")

# --- Civil & Residency Documents ---
# Certificate Number patterns
CERTIFICATE_NUMBER = re.compile(r"(?:certificate|cert|no|number)\s*[:#]?\s*([A-Z0-9\-/]{6,20})", re.IGNORECASE)
# Registration Number
REGISTRATION_NUMBER = re.compile(r"(?:registration|reg|serial)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]{6,20})", re.IGNORECASE)

# --- Land & Property Documents ---
# Survey Number
SURVEY_NUMBER = re.compile(r"(?:survey|s\.?no\.?|s.no)\s*[:#]?\s*([0-9\/\-]+)", re.IGNORECASE)
# Plot Number
PLOT_NUMBER = re.compile(r"(?:plot|p\.?no\.?|p.no)\s*[:#]?\s*([0-9\/\-]+)", re.IGNORECASE)

# --- Educational Documents ---
# Seat/Roll Number
SEAT_NUMBER = re.compile(r"(?:seat|roll|registration)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)
# Percentage
PERCENTAGE_PATTERN = re.compile(r"(\d{1,3}\.?\d{1,2})%", re.IGNORECASE)

# --- Category & Welfare Certificates ---
# Caste/Category
CASTE_PATTERN = re.compile(r"(?:caste|category|sub-caste)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# --- Other Patterns ---
# GST: 2 digits, 5 letters, 4 digits, 1 letter, 1 digit, 1 letter
GST_PATTERN = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ\d\b")
# Email
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
# Phone (Indian)
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
# Date (DD/MM/YYYY, DD-MM-YYYY, or DD.MM.YYYY)
DATE_PATTERN = re.compile(r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b")
# Year (YYYY)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Name patterns
NAME_PATTERN = re.compile(
    r"(?:full\s*name|name|applicant|student|owner)\s*[:]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
    re.IGNORECASE
)
NAME_FALLBACK = re.compile(r"\n([A-Z][A-Za-z ]{3,})\n")

# Gender
GENDER_PATTERN = re.compile(r"\b(MALE|FEMALE|Male|Female|M|F)\b")

# Address
ADDRESS_PATTERN = re.compile(r"(?:address|residence|permanent|present)\s*[:]?\s*(.*?)(?=\n|$)", re.IGNORECASE)

# Nationality
NATIONALITY_PATTERN = re.compile(r"nationality\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# Father's/Husband's Name
FATHER_NAME_PATTERN = re.compile(r"(?:father|husband|spouse)\s*['']?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# Mother's Name
MOTHER_NAME_PATTERN = re.compile(r"mother\s*['']?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# Invoice patterns
INVOICE_PATTERN = re.compile(r"invoice\s*(?:no|number)?\s*[:]?\s*([A-Za-z0-9\-/]+)", re.IGNORECASE)

# Total amount
TOTAL_PATTERN = re.compile(
    r"(?:total\s*amount|grand\s*total|amount\s*paid|total)\s*[:]?\s*[₹$]?\s*([\d,]+\.\d{2})",
    re.IGNORECASE
)

# Ration Card Number
RATION_CARD_PATTERN = re.compile(r"(?:ration|rc)\s*(?:card)?\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)

# Vehicle Number
VEHICLE_NUMBER_PATTERN = re.compile(r"(?:vehicle|registration|reg)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)

# Engine Number
ENGINE_NUMBER_PATTERN = re.compile(r"engine\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)

# Chassis Number
CHASSIS_NUMBER_PATTERN = re.compile(r"(?:chassis|frame|vin)\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE)

# Blood Group
BLOOD_GROUP_PATTERN = re.compile(r"blood\s*(?:group|type)?\s*[:]?\s*([A-Za-z+-]+)", re.IGNORECASE)

# District/Constituency
DISTRICT_PATTERN = re.compile(r"(?:district|constituency|taluka|tehsil)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# Village/Town
VILLAGE_PATTERN = re.compile(r"(?:village|town|city|taluka)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE)

# Income Amount
INCOME_PATTERN = re.compile(r"(?:income|annual|yearly)\s*[:]?\s*[₹$]?\s*([\d,]+\.?\d*)", re.IGNORECASE)

# Validity/Expiry
VALIDITY_PATTERN = re.compile(r"(?:valid|expiry|expiration|validity)\s*(?:up to|till|date)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE)

# Issue Date
ISSUE_DATE_PATTERN = re.compile(r"(?:issue|issued|date of issue)\s*(?:date)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE)

# ============================================================
#  TEXT CLEANING UTILITIES
# ============================================================
def clean_ocr_text(text: str) -> str:
    """Clean OCR text by removing special characters and normalizing whitespace."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_json_response(content: str) -> Dict:
    """Clean and parse JSON response from AI."""
    if not content:
        return {}
    try:
        content = re.sub(r"```json|```", "", content, flags=re.IGNORECASE).strip()
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return {}
        return json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON response", exc_info=True)
        return {}
    except Exception:
        logger.exception("Unexpected error in clean_json_response")
        return {}


def mask_sensitive_data(text: str) -> str:
    """Mask sensitive data like Aadhaar and PAN numbers."""
    if not text:
        return ""
    text = AADHAAR_PATTERN.sub("XXXX XXXX XXXX", text)
    text = PAN_PATTERN.sub("XXXXX0000X", text)
    text = VOTER_ID_PATTERN.sub("XXX0000000", text)
    text = PASSPORT_PATTERN.sub("X0000000", text)
    return text


def extract_field(text: str, pattern: Pattern, group: int = 1, default: str = "") -> Optional[str]:
    """Helper to extract a field using a regex pattern."""
    match = pattern.search(text)
    if match:
        return match.group(group).strip()
    return default if default is not None else None


# ============================================================
#  DOCUMENT TYPE DETECTION
# ============================================================
def detect_document_type(text: str) -> Optional[str]:
    """
    Detect document type from OCR text.
    Returns the document type as a string matching the model's DOC_TYPES.
    """
    if not text or not isinstance(text, str):
        return None
    
    cleaned = clean_ocr_text(text)
    if len(cleaned) < MIN_TEXT_LENGTH_FOR_DETECTION:
        return None
    
    cleaned_lower = cleaned.lower()
    
    # --- Identity Documents ---
    if "aadhaar" in cleaned_lower or "uidai" in cleaned_lower:
        return "aadhaar_card"
    
    if ("income tax" in cleaned_lower and "permanent account" in cleaned_lower) or \
       ("pan" in cleaned_lower and "department" in cleaned_lower):
        return "pan_card"
    
    if "voter id" in cleaned_lower or "election" in cleaned_lower or "epic" in cleaned_lower:
        return "voter_id_card"
    
    if "passport" in cleaned_lower and ("republic of india" in cleaned_lower or "india" in cleaned_lower):
        return "passport"
    
    if "driving" in cleaned_lower and ("licence" in cleaned_lower or "license" in cleaned_lower):
        return "driving_license"
    
    if "registration certificate" in cleaned_lower or "rc" in cleaned_lower or \
       ("vehicle" in cleaned_lower and "register" in cleaned_lower):
        return "vehicle_registration_certificate"
    
    # --- Civil & Residency Documents ---
    if "domicile" in cleaned_lower or "resident" in cleaned_lower:
        return "domicile_certificate"
    
    if "nationality" in cleaned_lower or "citizenship" in cleaned_lower:
        return "nationality_certificate"
    
    if "birth" in cleaned_lower and "certificate" in cleaned_lower:
        return "birth_certificate"
    
    if "marriage" in cleaned_lower and "certificate" in cleaned_lower:
        return "marriage_certificate"
    
    if "death" in cleaned_lower and "certificate" in cleaned_lower:
        return "death_certificate"
    
    # --- Land & Property Documents ---
    if "property card" in cleaned_lower or "7/12" in cleaned_lower or "8a" in cleaned_lower:
        return "property_card"
    
    if "income certificate" in cleaned_lower or "income proof" in cleaned_lower:
        return "income_certificate"
    
    if "ration card" in cleaned_lower or "ration" in cleaned_lower:
        return "ration_card"
    
    # --- Educational Documents ---
    if "school leaving" in cleaned_lower or "lc" in cleaned_lower or "transfer certificate" in cleaned_lower:
        return "school_leaving_certificate"
    
    if "ssc" in cleaned_lower and ("marksheet" in cleaned_lower or "marksheet" in cleaned_lower):
        return "ssc_marksheet"
    
    if "hsc" in cleaned_lower and ("marksheet" in cleaned_lower or "marksheet" in cleaned_lower):
        return "hsc_marksheet"
    
    if "degree" in cleaned_lower and "certificate" in cleaned_lower:
        return "degree_certificate"
    
    if "board" in cleaned_lower and "passing" in cleaned_lower:
        return "board_passing_certificate"
    
    # --- Category & Welfare Certificates ---
    if "caste" in cleaned_lower and "certificate" in cleaned_lower:
        if "validity" in cleaned_lower:
            return "caste_validity_certificate"
        return "caste_certificate"
    
    if "non creamy" in cleaned_lower or "ncl" in cleaned_lower:
        return "non_creamy_layer_certificate"
    
    if "ews" in cleaned_lower or "economically weaker" in cleaned_lower:
        return "ews_certificate"
    
    # --- Other Documents ---
    if "gst" in cleaned_lower and "certificate" in cleaned_lower:
        return "gst_certificate"
    
    if "invoice" in cleaned_lower or "bill" in cleaned_lower:
        return "invoice"
    
    return "other_document"


# ============================================================
#  DOCUMENT TYPE TO FIELD MAPPING
# ============================================================
DOC_TYPE_TO_EXTRACTOR = {
    "aadhaar_card": "_extract_aadhaar",
    "pan_card": "_extract_pan",
    "voter_id_card": "_extract_voter_id",
    "passport": "_extract_passport",
    "driving_license": "_extract_driving_license",
    "vehicle_registration_certificate": "_extract_vehicle_registration",
    "domicile_certificate": "_extract_domicile",
    "nationality_certificate": "_extract_nationality",
    "birth_certificate": "_extract_birth",
    "marriage_certificate": "_extract_marriage",
    "death_certificate": "_extract_death",
    "property_card": "_extract_property",
    "income_certificate": "_extract_income",
    "ration_card": "_extract_ration",
    "school_leaving_certificate": "_extract_school_leaving",
    "ssc_marksheet": "_extract_ssc",
    "hsc_marksheet": "_extract_hsc",
    "degree_certificate": "_extract_degree",
    "board_passing_certificate": "_extract_board_passing",
    "caste_certificate": "_extract_caste",
    "caste_validity_certificate": "_extract_caste_validity",
    "non_creamy_layer_certificate": "_extract_ncl",
    "ews_certificate": "_extract_ews",
    "gst_certificate": "_extract_gst",
    "invoice": "_extract_invoice",
}


# ============================================================
#  STRUCTURED EXTRACTION ENGINE
# ============================================================
def extract_structured_data(text: str) -> Dict[str, Any]:
    """
    Extract structured data from OCR text based on document type.
    Returns a dict of extracted fields.
    """
    if not text or not isinstance(text, str):
        return {}
    
    cleaned = clean_ocr_text(text)
    if len(cleaned) < MIN_TEXT_LENGTH_FOR_EXTRACTION:
        return {}
    
    doc_type = detect_document_type(cleaned)
    if not doc_type:
        return _generic_extraction(cleaned)
    
    # Get the appropriate extractor function
    extractor_name = DOC_TYPE_TO_EXTRACTOR.get(doc_type)
    if extractor_name:
        extractor = globals().get(extractor_name)
        if extractor:
            return extractor(cleaned)
    
    return _generic_extraction(cleaned)


# ============================================================
#  EXTRACTOR FUNCTIONS
# ============================================================

# ------------------------------------------------------------
#  IDENTITY DOCUMENTS
# ------------------------------------------------------------

def _extract_aadhaar(text: str) -> Dict[str, Any]:
    """Extract fields from Aadhaar Card."""
    data = {}
    
    # Aadhaar Number
    aadhaar = AADHAAR_PATTERN.search(text)
    if aadhaar:
        data["Aadhaar Number"] = aadhaar.group()
    
    # Full Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    # Date of Birth
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    # Gender
    gender = GENDER_PATTERN.search(text)
    if gender:
        data["Gender"] = gender.group()
    
    # Address
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    # QR Code (if present)
    if "qr" in text.lower():
        data["QR Code"] = "Present"
    
    return data


def _extract_pan(text: str) -> Dict[str, Any]:
    """Extract fields from PAN Card."""
    data = {}
    
    # PAN Number
    pan = PAN_PATTERN.search(text)
    if pan:
        data["PAN Number"] = pan.group()
    
    # Full Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    # Date of Birth
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    # Father's Name
    father = FATHER_NAME_PATTERN.search(text)
    if father:
        data["Father's Name"] = father.group(1).strip()
    
    # Date of Issue
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    return data


def _extract_voter_id(text: str) -> Dict[str, Any]:
    """Extract fields from Voter ID Card."""
    data = {}
    
    # Voter ID Number (EPIC)
    voter_id = VOTER_ID_PATTERN.search(text)
    if voter_id:
        data["Voter ID Number"] = voter_id.group()
    
    # Full Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    # Date of Birth
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    # Gender
    gender = GENDER_PATTERN.search(text)
    if gender:
        data["Gender"] = gender.group()
    
    # Father/Husband Name
    father = FATHER_NAME_PATTERN.search(text)
    if father:
        data["Father/Husband Name"] = father.group(1).strip()
    
    # Assembly Constituency
    district = DISTRICT_PATTERN.search(text)
    if district:
        data["Assembly Constituency"] = district.group(1).strip()
    
    # Address
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    return data


def _extract_passport(text: str) -> Dict[str, Any]:
    """Extract fields from Passport."""
    data = {}
    
    # Passport Number
    passport = PASSPORT_PATTERN.search(text)
    if passport:
        data["Passport Number"] = passport.group()
    
    # Full Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    # Date of Birth
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    # Place of Birth
    place = extract_field(text, re.compile(r"place of birth\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if place:
        data["Place of Birth"] = place
    
    # Nationality
    nationality = NATIONALITY_PATTERN.search(text)
    if nationality:
        data["Nationality"] = nationality.group(1).strip()
    
    # Date of Issue
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    # Expiry Date
    validity = VALIDITY_PATTERN.search(text)
    if validity:
        data["Expiry Date"] = validity.group(1).strip()
    
    # File Number
    file_no = extract_field(text, re.compile(r"file\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE))
    if file_no:
        data["File Number"] = file_no
    
    return data


def _extract_driving_license(text: str) -> Dict[str, Any]:
    """Extract fields from Driving License."""
    data = {}
    
    # License Number
    dl = DL_PATTERN.search(text)
    if dl:
        data["License Number"] = dl.group()
    
    # Full Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    # Date of Birth
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    # Blood Group
    blood = BLOOD_GROUP_PATTERN.search(text)
    if blood:
        data["Blood Group"] = blood.group(1).strip()
    
    # Address
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    # Date of Issue
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    # Valid Until
    validity = VALIDITY_PATTERN.search(text)
    if validity:
        data["Valid Until"] = validity.group(1).strip()
    
    # Vehicle Class
    vehicle_class = extract_field(text, re.compile(r"(?:vehicle|class)\s*(?:type)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if vehicle_class:
        data["Vehicle Class"] = vehicle_class
    
    return data


def _extract_vehicle_registration(text: str) -> Dict[str, Any]:
    """Extract fields from Vehicle Registration Certificate."""
    data = {}
    
    # Registration Number
    reg = VEHICLE_REG_PATTERN.search(text)
    if reg:
        data["Registration Number"] = reg.group()
    
    # Owner Name
    name = NAME_PATTERN.search(text)
    if name:
        data["Owner Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Owner Name"] = fallback.group(1).strip()
    
    # Vehicle Type
    vehicle_type = extract_field(text, re.compile(r"(?:vehicle|type|model)\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if vehicle_type:
        data["Vehicle Type"] = vehicle_type
    
    # Registration Date
    reg_date = DATE_PATTERN.search(text)
    if reg_date:
        data["Registration Date"] = reg_date.group()
    
    # Engine Number
    engine = ENGINE_NUMBER_PATTERN.search(text)
    if engine:
        data["Engine Number"] = engine.group(1).strip()
    
    # Chassis Number
    chassis = CHASSIS_NUMBER_PATTERN.search(text)
    if chassis:
        data["Chassis Number"] = chassis.group(1).strip()
    
    # Manufacturer
    manufacturer = extract_field(text, re.compile(r"(?:manufacturer|make)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if manufacturer:
        data["Vehicle Manufacturer"] = manufacturer
    
    # Fuel Type
    fuel = extract_field(text, re.compile(r"fuel\s*(?:type)?\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if fuel:
        data["Fuel Type"] = fuel
    
    # Color
    color = extract_field(text, re.compile(r"color\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if color:
        data["Color"] = color
    
    # Valid Until
    validity = VALIDITY_PATTERN.search(text)
    if validity:
        data["Valid Until"] = validity.group(1).strip()
    
    return data


# ------------------------------------------------------------
#  CIVIL & RESIDENCY DOCUMENTS
# ------------------------------------------------------------

def _extract_domicile(text: str) -> Dict[str, Any]:
    """Extract fields from Domicile Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    period = extract_field(text, re.compile(r"(?:period|duration)\s*(?:of residence)?\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if period:
        data["Period of Residence"] = period
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Certificate Number"] = cert.group(1).strip()
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    authority = extract_field(text, re.compile(r"(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if authority:
        data["Authority"] = authority
    
    return data


def _extract_nationality(text: str) -> Dict[str, Any]:
    """Extract fields from Nationality Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    nationality = NATIONALITY_PATTERN.search(text)
    if nationality:
        data["Citizenship Declaration"] = nationality.group(1).strip()
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Outward Number"] = cert.group(1).strip()
    
    authority = extract_field(text, re.compile(r"(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if authority:
        data["Issuing Authority"] = authority
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    return data


def _extract_birth(text: str) -> Dict[str, Any]:
    """Extract fields from Birth Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Full Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Full Name"] = fallback.group(1).strip()
    
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    place = extract_field(text, re.compile(r"place of birth\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if place:
        data["Place of Birth"] = place
    
    mother = MOTHER_NAME_PATTERN.search(text)
    if mother:
        data["Mother's Name"] = mother.group(1).strip()
    
    father = FATHER_NAME_PATTERN.search(text)
    if father:
        data["Father's Name"] = father.group(1).strip()
    
    reg = REGISTRATION_NUMBER.search(text)
    if reg:
        data["Registration Number"] = reg.group(1).strip()
    
    reg_date = extract_field(text, re.compile(r"date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if reg_date:
        data["Date of Registration"] = reg_date
    
    return data


def _extract_marriage(text: str) -> Dict[str, Any]:
    """Extract fields from Marriage Certificate."""
    data = {}
    
    husband = FATHER_NAME_PATTERN.search(text)
    if husband:
        data["Husband's Name"] = husband.group(1).strip()
    
    wife = extract_field(text, re.compile(r"wife\s*['']?s?\s*(?:name)?\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if wife:
        data["Wife's Name"] = wife
    
    marriage_date = DATE_PATTERN.search(text)
    if marriage_date:
        data["Date of Marriage"] = marriage_date.group()
    
    place = extract_field(text, re.compile(r"place of marriage\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if place:
        data["Place of Marriage"] = place
    
    reg = REGISTRATION_NUMBER.search(text)
    if reg:
        data["Registration Number"] = reg.group(1).strip()
    
    reg_date = extract_field(text, re.compile(r"date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if reg_date:
        data["Date of Registration"] = reg_date
    
    return data


def _extract_death(text: str) -> Dict[str, Any]:
    """Extract fields from Death Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Deceased Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Deceased Name"] = fallback.group(1).strip()
    
    death_date = DATE_PATTERN.search(text)
    if death_date:
        data["Date of Death"] = death_date.group()
    
    place = extract_field(text, re.compile(r"place of death\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if place:
        data["Place of Death"] = place
    
    cause = extract_field(text, re.compile(r"cause of death\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if cause:
        data["Cause of Death"] = cause
    
    reg = REGISTRATION_NUMBER.search(text)
    if reg:
        data["Registration Number"] = reg.group(1).strip()
    
    reg_date = extract_field(text, re.compile(r"date of registration\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if reg_date:
        data["Date of Registration"] = reg_date
    
    return data


# ------------------------------------------------------------
#  LAND & PROPERTY DOCUMENTS
# ------------------------------------------------------------

def _extract_property(text: str) -> Dict[str, Any]:
    """Extract fields from Property Card."""
    data = {}
    
    survey = SURVEY_NUMBER.search(text)
    if survey:
        data["Survey Number"] = survey.group(1).strip()
    
    plot = PLOT_NUMBER.search(text)
    if plot:
        data["Plot Number"] = plot.group(1).strip()
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Owner Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Owner Name"] = fallback.group(1).strip()
    
    area = extract_field(text, re.compile(r"(?:area|land)\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if area:
        data["Land Area"] = area
    
    village = VILLAGE_PATTERN.search(text)
    if village:
        data["Village Name"] = village.group(1).strip()
    
    tax = extract_field(text, re.compile(r"(?:tax|assessment)\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if tax:
        data["Tax Assessment Details"] = tax
    
    return data


def _extract_income(text: str) -> Dict[str, Any]:
    """Extract fields from Income Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    income = INCOME_PATTERN.search(text)
    if income:
        data["Annual Family Income"] = income.group(1).strip()
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Financial Year"] = year.group()
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Certificate Number"] = cert.group(1).strip()
    
    authority = extract_field(text, re.compile(r"(?:authority|issuing|by)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if authority:
        data["Authority"] = authority
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    return data


def _extract_ration(text: str) -> Dict[str, Any]:
    """Extract fields from Ration Card."""
    data = {}
    
    ration = RATION_CARD_PATTERN.search(text)
    if ration:
        data["Ration Card Number"] = ration.group(1).strip()
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Household Head Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Household Head Name"] = fallback.group(1).strip()
    
    members = extract_field(text, re.compile(r"(?:members|family)\s*(?:members)?\s*[:]?\s*(\d+)", re.IGNORECASE))
    if members:
        data["Family Member Count"] = members
    
    category = extract_field(text, re.compile(r"(?:category|type|color)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if category:
        data["Category"] = category
    
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    return data


# ------------------------------------------------------------
#  EDUCATIONAL DOCUMENTS
# ------------------------------------------------------------

def _extract_school_leaving(text: str) -> Dict[str, Any]:
    """Extract fields from School Leaving Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Student Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Student Name"] = fallback.group(1).strip()
    
    dob = DATE_PATTERN.search(text)
    if dob:
        data["Date of Birth"] = dob.group()
    
    place = extract_field(text, re.compile(r"place of birth\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if place:
        data["Place of Birth"] = place
    
    religion = extract_field(text, re.compile(r"religion\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if religion:
        data["Religion"] = religion
    
    caste = CASTE_PATTERN.search(text)
    if caste:
        data["Caste"] = caste.group(1).strip()
    
    leaving_date = DATE_PATTERN.search(text)
    if leaving_date:
        data["Date of Leaving"] = leaving_date.group()
    
    school = extract_field(text, re.compile(r"(?:school|institution)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if school:
        data["School Name"] = school
    
    return data


def _extract_ssc(text: str) -> Dict[str, Any]:
    """Extract fields from SSC Marksheet."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Student Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Student Name"] = fallback.group(1).strip()
    
    seat = SEAT_NUMBER.search(text)
    if seat:
        data["Seat Number"] = seat.group(1).strip()
    
    percentage = PERCENTAGE_PATTERN.search(text)
    if percentage:
        data["Total Percentage"] = percentage.group(1) + "%"
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Year"] = year.group()
    
    division = extract_field(text, re.compile(r"(?:division|grade|class)\s*[:]?\s*([A-Za-z0-9 ]+)", re.IGNORECASE))
    if division:
        data["Division/Grade"] = division
    
    return data


def _extract_hsc(text: str) -> Dict[str, Any]:
    """Extract fields from HSC Marksheet."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Student Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Student Name"] = fallback.group(1).strip()
    
    seat = SEAT_NUMBER.search(text)
    if seat:
        data["Seat Number"] = seat.group(1).strip()
    
    percentage = PERCENTAGE_PATTERN.search(text)
    if percentage:
        data["Total Percentage"] = percentage.group(1) + "%"
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Year"] = year.group()
    
    division = extract_field(text, re.compile(r"(?:division|grade|class)\s*[:]?\s*([A-Za-z0-9 ]+)", re.IGNORECASE))
    if division:
        data["Division/Grade"] = division
    
    return data


def _extract_degree(text: str) -> Dict[str, Any]:
    """Extract fields from Degree Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Student Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Student Name"] = fallback.group(1).strip()
    
    serial = CERTIFICATE_NUMBER.search(text)
    if serial:
        data["Degree Serial Number"] = serial.group(1).strip()
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Passing Year"] = year.group()
    
    course = extract_field(text, re.compile(r"(?:course|degree|program)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if course:
        data["Course Name"] = course
    
    classification = extract_field(text, re.compile(r"(?:classification|class|grade)\s*[:]?\s*([A-Za-z0-9 ]+)", re.IGNORECASE))
    if classification:
        data["Classification/Class"] = classification
    
    university = extract_field(text, re.compile(r"(?:university|institution)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if university:
        data["University Name"] = university
    
    return data


def _extract_board_passing(text: str) -> Dict[str, Any]:
    """Extract fields from Board Passing Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Student Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Student Name"] = fallback.group(1).strip()
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Passing Year"] = year.group()
    
    board = extract_field(text, re.compile(r"(?:board|university)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if board:
        data["Board/University Name"] = board
    
    total = extract_field(text, re.compile(r"(?:total|obtained)\s*(?:marks)?\s*[:]?\s*([0-9/ ]+)", re.IGNORECASE))
    if total:
        data["Total Marks Obtained"] = total
    
    result = extract_field(text, re.compile(r"(?:result|status)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if result:
        data["Result Status"] = result
    
    return data


# ------------------------------------------------------------
#  CATEGORY & WELFARE CERTIFICATES
# ------------------------------------------------------------

def _extract_caste(text: str) -> Dict[str, Any]:
    """Extract fields from Caste Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    caste = CASTE_PATTERN.search(text)
    if caste:
        data["Caste Category"] = caste.group(1).strip()
    
    sub_caste = extract_field(text, re.compile(r"sub[-\s]caste\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if sub_caste:
        data["Sub-Caste Name"] = sub_caste
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Outward Number"] = cert.group(1).strip()
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    return data


def _extract_caste_validity(text: str) -> Dict[str, Any]:
    """Extract fields from Caste Validity Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    decision = extract_field(text, re.compile(r"(?:decision|verdict)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if decision:
        data["Scrutiny Committee Decision"] = decision
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Validity Certificate Number"] = cert.group(1).strip()
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    case = extract_field(text, re.compile(r"case\s*(?:no|number)?\s*[:#]?\s*([A-Z0-9\-/]+)", re.IGNORECASE))
    if case:
        data["Case Number"] = case
    
    return data


def _extract_ncl(text: str) -> Dict[str, Any]:
    """Extract fields from Non-Creamy Layer Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    year = YEAR_PATTERN.search(text)
    if year:
        data["Financial Year"] = year.group()
    
    sub_caste = extract_field(text, re.compile(r"sub[-\s]caste\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if sub_caste:
        data["OBC Sub-Caste"] = sub_caste
    
    validity = VALIDITY_PATTERN.search(text)
    if validity:
        data["Validity Expiry Date"] = validity.group(1).strip()
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Certificate Number"] = cert.group(1).strip()
    
    return data


def _extract_ews(text: str) -> Dict[str, Any]:
    """Extract fields from EWS Certificate."""
    data = {}
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Applicant Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Applicant Name"] = fallback.group(1).strip()
    
    assets = extract_field(text, re.compile(r"(?:asset|property|valuation)\s*[:]?\s*([A-Za-z0-9\-/ ]+)", re.IGNORECASE))
    if assets:
        data["Asset Valuation Details"] = assets
    
    income = INCOME_PATTERN.search(text)
    if income:
        data["Annual Income Limit Verification"] = income.group(1).strip()
    
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Certificate Number"] = cert.group(1).strip()
    
    issue = ISSUE_DATE_PATTERN.search(text)
    if issue:
        data["Date of Issue"] = issue.group(1).strip()
    
    return data


# ------------------------------------------------------------
#  OTHER DOCUMENTS
# ------------------------------------------------------------

def _extract_gst(text: str) -> Dict[str, Any]:
    """Extract fields from GST Certificate."""
    data = {}
    
    gst = GST_PATTERN.search(text)
    if gst:
        data["GST Number"] = gst.group()
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Business Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Business Name"] = fallback.group(1).strip()
    
    business_type = extract_field(text, re.compile(r"(?:business|type)\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if business_type:
        data["Business Type"] = business_type
    
    reg_date = DATE_PATTERN.search(text)
    if reg_date:
        data["Date of Registration"] = reg_date.group()
    
    state = DISTRICT_PATTERN.search(text)
    if state:
        data["State"] = state.group(1).strip()
    
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    status = extract_field(text, re.compile(r"status\s*[:]?\s*([A-Za-z ]+)", re.IGNORECASE))
    if status:
        data["Status"] = status
    
    return data


def _extract_invoice(text: str) -> Dict[str, Any]:
    """Extract fields from Invoice."""
    data = {}
    
    invoice = INVOICE_PATTERN.search(text)
    if invoice:
        data["Invoice Number"] = invoice.group(1).strip()
    
    inv_date = DATE_PATTERN.search(text)
    if inv_date:
        data["Invoice Date"] = inv_date.group()
    
    customer = NAME_PATTERN.search(text)
    if customer:
        data["Customer Name"] = customer.group(1).strip()
    
    customer_gst = GST_PATTERN.search(text)
    if customer_gst:
        data["Customer GST"] = customer_gst.group()
    
    total = TOTAL_PATTERN.search(text)
    if total:
        data["Total Amount"] = total.group(1).strip()
    
    tax = extract_field(text, re.compile(r"(?:tax|gst|vat)\s*(?:amount)?\s*[:]?\s*[₹$]?\s*([\d,]+\.?\d*)", re.IGNORECASE))
    if tax:
        data["Tax Amount"] = tax
    
    return data

# ============================================================
#  GENERIC EXTRACTION (for fallback)
# ============================================================
def generic_extraction(text: str) -> Dict[str, Any]:
    """
    Generic extraction for unknown document types.
    This is a fallback function used when specific extractors aren't available.
    """
    if not text or not isinstance(text, str):
        return {}
    
    data = {}
    
    email = EMAIL_PATTERN.search(text)
    if email:
        data["Email"] = email.group()
    
    phone = PHONE_PATTERN.search(text)
    if phone:
        data["Phone"] = phone.group()
    
    name = NAME_PATTERN.search(text)
    if name:
        data["Name"] = name.group(1).strip()
    else:
        fallback = NAME_FALLBACK.search(text)
        if fallback:
            data["Name"] = fallback.group(1).strip()
    
    date = DATE_PATTERN.search(text)
    if date:
        data["Date"] = date.group()
    
    ref = CERTIFICATE_NUMBER.search(text)
    if ref:
        data["Reference Number"] = ref.group(1).strip()
    
    return data

# ============================================================
#  MERGE MULTI‑PAGE DATA
# ============================================================
def merge_page_data(pages: List[Dict]) -> Dict[str, Any]:
    """Merge data from multiple pages."""
    merged = {}
    for page in pages:
        for key, value in page.items():
            if key == "_metadata":
                continue
            if key not in merged:
                merged[key] = value
            elif isinstance(merged[key], list) and isinstance(value, list):
                merged[key].extend(value)
            elif isinstance(merged[key], str) and isinstance(value, str):
                merged[key] = f"{merged[key]} {value}"
    return merged


# ============================================================
#  VALIDATION FUNCTIONS
# ============================================================
def validate_aadhaar(aadhaar: str) -> bool:
    """Validate Aadhaar number format."""
    return bool(AADHAAR_PATTERN.match(aadhaar))


def validate_pan(pan: str) -> bool:
    """Validate PAN number format."""
    return bool(PAN_PATTERN.match(pan))


def validate_voter_id(voter_id: str) -> bool:
    """Validate Voter ID format."""
    return bool(VOTER_ID_PATTERN.match(voter_id))


def validate_passport(passport: str) -> bool:
    """Validate Passport number format."""
    return bool(PASSPORT_PATTERN.match(passport))


def validate_dl(dl: str) -> bool:
    """Validate Driving License number format."""
    return bool(DL_PATTERN.match(dl))


def validate_gst(gst: str) -> bool:
    """Validate GST number format."""
    return bool(GST_PATTERN.match(gst))


def validate_phone(phone: str) -> bool:
    """Validate Indian phone number format."""
    return bool(PHONE_PATTERN.match(phone))


def validate_email(email: str) -> bool:
    """Validate email format."""
    return bool(EMAIL_PATTERN.match(email))


def validate_date(date: str) -> bool:
    """Validate date format (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)."""
    return bool(DATE_PATTERN.match(date))


# ============================================================
#  EXTRACT ALL VALID FIELDS
# ============================================================
def extract_all_fields(text: str) -> Dict[str, Any]:
    """Extract all possible fields from text."""
    data = {}
    
    # Identity Numbers
    aadhaar = AADHAAR_PATTERN.search(text)
    if aadhaar:
        data["Aadhaar Number"] = aadhaar.group()
    
    pan = PAN_PATTERN.search(text)
    if pan:
        data["PAN Number"] = pan.group()
    
    voter_id = VOTER_ID_PATTERN.search(text)
    if voter_id:
        data["Voter ID Number"] = voter_id.group()
    
    passport = PASSPORT_PATTERN.search(text)
    if passport:
        data["Passport Number"] = passport.group()
    
    dl = DL_PATTERN.search(text)
    if dl:
        data["License Number"] = dl.group()
    
    gst = GST_PATTERN.search(text)
    if gst:
        data["GST Number"] = gst.group()
    
    # Personal Info
    name = NAME_PATTERN.search(text)
    if name:
        data["Name"] = name.group(1).strip()
    
    father = FATHER_NAME_PATTERN.search(text)
    if father:
        data["Father's Name"] = father.group(1).strip()
    
    mother = MOTHER_NAME_PATTERN.search(text)
    if mother:
        data["Mother's Name"] = mother.group(1).strip()
    
    email = EMAIL_PATTERN.search(text)
    if email:
        data["Email"] = email.group()
    
    phone = PHONE_PATTERN.search(text)
    if phone:
        data["Phone"] = phone.group()
    
    # Address
    addr = ADDRESS_PATTERN.search(text)
    if addr:
        data["Address"] = addr.group(1).strip()
    
    # Dates
    date = DATE_PATTERN.search(text)
    if date:
        data["Date"] = date.group()
    
    # Certificates
    cert = CERTIFICATE_NUMBER.search(text)
    if cert:
        data["Certificate Number"] = cert.group(1).strip()
    
    return data