import os
import cv2
import pytesseract
from django.conf import settings
from pdf2image import convert_from_path
import openbharatocr

# Set Tesseract command path (from settings or default)
pytesseract.pytesseract.tesseract_cmd = getattr(
    settings, "PYTESSERACT_CMD", "tesseract"
)

# Map document types to OpenBharatOCR functions
OCR_HANDLERS = {
    "pan": openbharatocr.pan,
    "aadhaar_front": openbharatocr.front_aadhaar,
    "aadhaar_back": openbharatocr.back_aadhaar,
    "dl": openbharatocr.driving_licence,
    "passport": openbharatocr.passport,
    "voter_front": openbharatocr.voter_id_front,
    "voter_back": openbharatocr.voter_id_back,
    "rc": openbharatocr.vehicle_registration,
}


def preprocess_image(image_path: str) -> "cv2.Mat":
    """Preprocess image for OCR fallback using pytesseract."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def pdf_to_images(pdf_path: str) -> list:
    """Convert a PDF to a list of images (one per page)."""
    images = convert_from_path(pdf_path, dpi=300)
    if not images:
        raise ValueError("PDF has no pages")

    media_folder = os.path.join(settings.MEDIA_ROOT, "documents")
    os.makedirs(media_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    img_paths = []

    for i, img in enumerate(images, start=1):
        img_filename = f"{base_name}_page{i}.jpg"
        img_path = os.path.join(media_folder, img_filename)
        img.save(img_path, "JPEG")
        img_paths.append(img_path)

    return img_paths


def process_document_file(file_path: str, doc_type: str = None) -> dict:
    """
    Process a document (image or PDF) and extract structured fields.

    Returns a dictionary per page:
    {
        "page_1": { "Field1": "Value1", "Field2": "Value2", ... },
        "page_2": { ... }
    }

    - Uses OpenBharatOCR if available for the document type.
    - Falls back to Tesseract OCR if OpenBharatOCR fails.
    """
    ext = os.path.splitext(file_path)[1].lower()
    image_paths = [file_path]

    # Convert PDF to images if needed
    if ext == ".pdf":
        image_paths = pdf_to_images(file_path)

    extracted_results = {}

    for idx, path in enumerate(image_paths, start=1):
        page_key = f"page_{idx}"
        result = {}

        # Try OpenBharatOCR for structured extraction
        if doc_type in OCR_HANDLERS:
            try:
                structured_data = OCR_HANDLERS[doc_type](path)
                if isinstance(structured_data, dict) and structured_data:
                    result = structured_data
            except Exception as e:
                print(f"[Warning] OpenBharatOCR failed on {page_key}: {e}")

        # Fallback to Tesseract if OpenBharatOCR failed
        if not result:
            processed_img = preprocess_image(path)
            text = pytesseract.image_to_string(processed_img, lang="eng").strip()
            result = {"raw_text": text}

        # Remove raw_text if other structured fields exist
        if "raw_text" in result and len(result) > 1:
            del result["raw_text"]

        extracted_results[page_key] = result

    return extracted_results


def process_document_file(file_path: str, doc_type: str = None) -> dict:
    """
    Process a document (image or PDF) and extract structured fields.

    Returns a dictionary per page:
    {
        "page_1": { "Field1": "Value1", "Field2": "Value2", ... },
        "page_2": { ... }
    }

    - Uses OpenBharatOCR if available for the document type.
    - Falls back to Tesseract OCR if OpenBharatOCR fails.
    """
    import json

    ext = os.path.splitext(file_path)[1].lower()
    image_paths = [file_path]

    # Convert PDF to images if needed
    if ext == ".pdf":
        image_paths = pdf_to_images(file_path)

    extracted_results = {}

    for idx, path in enumerate(image_paths, start=1):
        page_key = f"page_{idx}"
        result = {}

        # Try OpenBharatOCR for structured extraction
        if doc_type in OCR_HANDLERS:
            try:
                structured_data = OCR_HANDLERS[doc_type](path)

                # Ensure we always have a dict
                if isinstance(structured_data, str):
                    try:
                        structured_data = json.loads(structured_data)
                    except json.JSONDecodeError:
                        structured_data = {"raw_text": structured_data}

                if isinstance(structured_data, dict) and structured_data:
                    result = structured_data
            except Exception as e:
                print(f"[Warning] OpenBharatOCR failed on {page_key}: {e}")

        # Fallback to Tesseract if OpenBharatOCR failed
        if not result:
            processed_img = preprocess_image(path)
            text = pytesseract.image_to_string(processed_img, lang="eng").strip()
            result = {"raw_text": text}

        # Remove raw_text if other structured fields exist
        if "raw_text" in result and len(result) > 1:
            del result["raw_text"]

        extracted_results[page_key] = result

    return extracted_results
