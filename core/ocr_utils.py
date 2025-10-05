import os
import cv2
import pytesseract
import numpy as np
from django.conf import settings
from pdf2image import convert_from_path
import openbharatocr
import json
from PIL import Image, ImageEnhance

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


# -------------------------------------------------
# 🔹 IMAGE QUALITY CHECK
# -------------------------------------------------
def is_image_blurry(image_path, threshold=100):
    """Detects blur using variance of Laplacian."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    fm = cv2.Laplacian(img, cv2.CV_64F).var()
    return fm < threshold  # lower = more blurry


# -------------------------------------------------
# 🔹 IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image(image_path: str) -> "cv2.Mat":
    """Enhance and clean the image for better OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Enhance contrast using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    return sharpened


# -------------------------------------------------
# 🔹 PDF TO IMAGE CONVERSION
# -------------------------------------------------
def pdf_to_images(pdf_path: str) -> list:
    """Convert PDF to image pages."""
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


# -------------------------------------------------
# 🔹 MAIN OCR PIPELINE
# -------------------------------------------------
def process_document_file(file_path: str, doc_type: str = None) -> dict:
    """
    Extract structured data from image/PDF document.
    Uses OpenBharatOCR if available; otherwise falls back to Tesseract.
    Handles blur and low-quality documents automatically.
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

        # ---- Check for Blur ----
        if is_image_blurry(path):
            extracted_results[page_key] = {
                "error": "Document too blurry or unclear. Please upload a sharper image."
            }
            continue

        # ---- Try OpenBharatOCR ----
        if doc_type in OCR_HANDLERS:
            try:
                structured_data = OCR_HANDLERS[doc_type](path)

                if isinstance(structured_data, str):
                    try:
                        structured_data = json.loads(structured_data)
                    except json.JSONDecodeError:
                        structured_data = {"raw_text": structured_data}

                if isinstance(structured_data, dict) and structured_data:
                    result = structured_data
            except Exception as e:
                print(f"[Warning] OpenBharatOCR failed on {page_key}: {e}")

        # ---- Fallback: Tesseract ----
        if not result:
            processed_img = preprocess_image(path)
            text = pytesseract.image_to_string(processed_img, lang="eng").strip()
            result = {"raw_text": text if text else "Unreadable or blank page"}

        # Clean up redundant field
        if "raw_text" in result and len(result) > 1:
            del result["raw_text"]

        extracted_results[page_key] = result

    return extracted_results
