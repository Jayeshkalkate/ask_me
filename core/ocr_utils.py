# C:\chatbot\ask_me\core\ocr_utils.py

import os
import cv2
import pytesseract
import numpy as np
from django.conf import settings
from pdf2image import convert_from_path
import json
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, List, Optional, Union, Tuple
import re
import tempfile
from skimage import exposure
import imutils
import sys
# 🔐 Safe OpenBharatOCR import
try:
    import openbharatocr
except ImportError:
    openbharatocr = None

# Configure logging
logger = logging.getLogger(__name__)

print("🔥 OCR_UTILS LOADED")

# =============================================================================
# 🔐 SAFE TESSERACT CONFIGURATION (MANDATORY)
# =============================================================================

TESSERACT_PATH = getattr(settings, "PYTESSERACT_CMD", None)

if not TESSERACT_PATH:
    raise RuntimeError("❌ PYTESSERACT_CMD not defined in Django settings.py")

if not os.path.exists(TESSERACT_PATH):
    raise RuntimeError(f"❌ Tesseract executable not found at: {TESSERACT_PATH}")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

logger.info(f"✅ Tesseract configured: {TESSERACT_PATH}")


# =============================================================================
# Custom Exceptions
# =============================================================================
class OCRException(Exception):
    """Base exception for OCR related errors."""

    pass


class ImageLoadError(OCRException):
    """Raised when an image cannot be loaded or is invalid."""

    pass


class PDFConversionError(OCRException):
    """Raised when PDF to image conversion fails."""

    pass


# =============================================================================
# Strict Image Loader
# =============================================================================
def load_image_strict(image_path: str) -> np.ndarray:
    """
    Strictly load an image from disk.
    Raises ImageLoadError if the file doesn't exist, can't be read, or has invalid dimensions.
    """
    if not os.path.exists(image_path):
        raise ImageLoadError(f"Image file not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ImageLoadError(
            f"OpenCV failed to read image (corrupt or unsupported format): {image_path}"
        )

    if img.size == 0:
        raise ImageLoadError(f"Image has zero pixels: {image_path}")

    return img


# =============================================================================
# Safe Tesseract Wrapper with Confidence Filtering
# =============================================================================
def run_tesseract_safe(
    image: np.ndarray,
    config: str = "--oem 3 --psm 6",
    lang: str = "eng",
    confidence_threshold: int = 60,
) -> str:

    if image is None or image.size == 0:
        logger.warning("Empty image passed to Tesseract")
        return ""

    try:
        data = pytesseract.image_to_data(
            image, config=config, lang=lang, output_type=pytesseract.Output.DICT
        )

        filtered_words = []

        for text, conf in zip(data["text"], data["conf"]):
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                continue

            text = text.strip()
            if conf >= confidence_threshold and text:
                filtered_words.append(text)

        return " ".join(filtered_words)

    except Exception as e:
        logger.error("Tesseract OCR failed", exc_info=True)
        return ""

# =============================================================================
# OCR Text Cleaner
# =============================================================================
def clean_ocr_text(text: str) -> str:
    """
    Clean OCR output: remove excess whitespace, fix common artifacts.
    """
    if not text:
        return ""
    # Replace multiple newlines/spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Remove stray characters (optional, can be expanded)
    text = re.sub(r"[^\w\s\/\-:,\.\(\)]", "", text)
    return text.strip()


# =============================================================================
# (Keep existing OCRPreprocessor, DocumentAnalyzer, is_image_blurry, etc.,
#  but replace cv2.imread with load_image_strict where appropriate)
# =============================================================================


class OCRPreprocessor:
    """Advanced image preprocessing for OCR optimization"""

    # ... (unchanged, but internal methods that use cv2.imread should use load_image_strict if they receive a path)
    # For now we keep them as they operate on already loaded images or receive paths.
    # We'll modify the functions that call them to use load_image_strict first.
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        # ... unchanged
        pass

    @staticmethod
    def enhance_resolution(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        # ... unchanged
        pass

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        # ... unchanged
        pass


class DocumentAnalyzer:
    """Analyze document quality and characteristics"""

    @staticmethod
    def calculate_image_quality_score(image_path: str) -> Dict[str, float]:
        try:
            img = load_image_strict(
                image_path
            )  # replaced safe_read_image with load_image_strict
        except ImageLoadError:
            return {
                "overall_score": 0.0,
                "blur_score": 0.0,
                "contrast_score": 0.0,
                "brightness_score": 0.0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast_score = np.std(gray)
        brightness_score = np.mean(gray)

        blur_normalized = min(blur_score / 1000, 1.0)
        contrast_normalized = min(contrast_score / 64, 1.0)
        brightness_normalized = 1 - abs(brightness_score - 127) / 127
        brightness_normalized = max(0.0, min(brightness_normalized, 1.0))

        overall_score = (
            blur_normalized + contrast_normalized + brightness_normalized
        ) / 3

        return {
            "overall_score": float(overall_score),
            "blur_score": float(blur_normalized),
            "contrast_score": float(contrast_normalized),
            "brightness_score": float(brightness_normalized),
        }

    @staticmethod
    def detect_document_type_from_text(text: str) -> str:
        # ... unchanged
        pass


def is_image_blurry(image_path: str, threshold: float = 100.0) -> Tuple[bool, float]:
    """Enhanced blur detection with multiple metrics"""
    try:
        img = load_image_strict(image_path)  # use strict loader
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except ImageLoadError:
        return True, 0.0

    # Method 1: Variance of Laplacian
    fm_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Method 2: FFT-based blur detection
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    # Calculate high frequency content
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    # Remove center region (low frequencies)
    fft_shift[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
    fft_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(fft_ishift)
    img_back = np.abs(img_back)

    high_freq_content = np.mean(img_back)

    # Combined blur score
    blur_score = (fm_laplacian + high_freq_content) / 2

    return blur_score < threshold, blur_score


def preprocess_image_advanced(image_path: str) -> np.ndarray:
    """
    Stable preprocessing for OCR.
    No over-processing.
    """
    try:
        img = load_image_strict(image_path)  # use strict loader
    except ImageLoadError as e:
        logger.error(f"Image load failed in preprocess_image_advanced: {e}")
        raise ImageLoadError("Preprocessing failed")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize (improves OCR accuracy)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Light denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold (SAFE)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return thresh


# -------------------------------------------------
# 🔹 ENHANCED PDF TO IMAGE CONVERSION (with validation)
# -------------------------------------------------
def pdf_to_images_enhanced(pdf_path: str, dpi: int = 300, poppler_path: str = None):
    try:
        if not os.path.exists(pdf_path):
            raise PDFConversionError(f"PDF file not found: {pdf_path}")

        if poppler_path and os.path.exists(poppler_path):
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path, dpi=dpi)

        if not images:
            raise PDFConversionError(
                "PDF has no pages or conversion produced empty result"
            )

        temp_dir = tempfile.mkdtemp()
        img_paths = []

        for i, img in enumerate(images, 1):
            img_path = os.path.join(temp_dir, f"page_{i}.jpg")
            img.save(img_path, "JPEG", quality=95)
            img_paths.append(img_path)

        return img_paths

    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise PDFConversionError(f"Failed to convert PDF to images: {e}") from e


# -------------------------------------------------
# 🔹 SINGLE OCR EXTRACTION (UPDATED WITH SAFE WRAPPER)
# -------------------------------------------------
def extract_text_with_tesseract(image_path: str) -> str:
    """
    Extract text using Tesseract OCR only.
    Uses strict image loading and safe Tesseract wrapper.
    """

    try:
        # -----------------------------------------
        # STEP 1 — STRICT LOAD
        # -----------------------------------------
        img = load_image_strict(image_path)

        # -----------------------------------------
        # STEP 2 — PREPROCESS
        # -----------------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # -----------------------------------------
        # STEP 3 — THRESHOLD
        # -----------------------------------------
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # =========================================
        # 🔴 VERY IMPORTANT DEBUG VALIDATION
        # =========================================
        if thresh is None or thresh.size == 0:
            logger.error("❌ Threshold image invalid before OCR")
            raise OCRException("Threshold image invalid")

        # Optional extra debug (recommended)
        if thresh.shape[0] < 10 or thresh.shape[1] < 10:
            logger.warning("⚠ Threshold image too small for OCR")

        # -----------------------------------------
        # STEP 4 — OCR CALL
        # -----------------------------------------
        config = "--oem 3 --psm 6 -l eng"
        text = run_tesseract_safe(
            thresh,
            config=config,
            lang="eng",
            confidence_threshold=60,
        )

        return clean_ocr_text(text)

    except ImageLoadError as e:
        logger.error(f"Image load failed in extract_text_with_tesseract: {e}")
        return ""

    except OCRException as e:
        logger.error(f"OCR validation failed: {e}")
        return ""

    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


# -------------------------------------------------
# 🔹 SMART OCR EXTRACTION (NOW JUST WRAPPER)
# -------------------------------------------------

def smart_ocr_extraction(image_path: str, doc_type: str = None) -> Dict:
    """
    Improved OCR extraction - SINGLE OCR CALL ONLY.
    Extracts raw OCR text once and optionally performs structured extraction.
    """

    try:
        logger.info("🚀 Starting OCR extraction")

        # -----------------------------------------
        # STEP 1 — SINGLE OCR CALL
        # -----------------------------------------
        
        text = extract_text_with_tesseract(image_path)

        if not text or len(text.strip()) < 10:
            logger.warning("⚠ OCR returned very little text")
            return {"status": "failed", "raw_text": text or ""}

        logger.info("✅ OCR successful")

        # -----------------------------------------
        # STEP 2 — STRUCTURED EXTRACTION (OPTIONAL)
        # -----------------------------------------
        structured_result = None

        if doc_type and doc_type in OCR_HANDLERS and openbharatocr:
            try:
                logger.info(f"📄 Running structured extraction for {doc_type}")
                structured_result = OCR_HANDLERS[doc_type](image_path)

            except Exception as e:
                logger.warning(
                    "⚠ OpenBharatOCR structured extraction failed", exc_info=True
                )
                structured_result = None

        # -----------------------------------------
        # STEP 3 — FINAL RESPONSE
        # -----------------------------------------
        return {
            "status": "success",
            "raw_text": text,
            "structured_data": structured_result,
        }

    except Exception as e:
        logger.error(f"❌ OCR failed: {e}", exc_info=True)
        return {"status": "failed", "raw_text": ""}


# -------------------------------------------------
# 🔹 ENHANCED MAIN OCR PIPELINE
# -------------------------------------------------
def process_document_file_enhanced(
    file_path: str, doc_type: str = None, auto_detect: bool = True
) -> Dict:

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    image_paths = [file_path]

    # -----------------------------------------
    # PDF → IMAGES
    # -----------------------------------------
    if ext == ".pdf":
        try:
            image_paths = pdf_to_images_enhanced(file_path)
        except PDFConversionError as e:
            return {"error": f"PDF conversion failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error during PDF conversion: {str(e)}"}

    extracted_results = {}
    analyzer = DocumentAnalyzer()

    # -----------------------------------------
    # PROCESS EACH PAGE - OCR RUNS ONLY ONCE PER PAGE
    # -----------------------------------------
    for idx, image_path in enumerate(image_paths, start=1):
        page_key = f"page_{idx}"

        try:
            # -----------------------------------------
            # STEP 0 — QUALITY CHECK
            # -----------------------------------------
            quality_scores = analyzer.calculate_image_quality_score(image_path)
            is_blurry, blur_score = is_image_blurry(image_path)

            # -----------------------------------------
            # STEP 1 — RUN OCR (SINGLE CALL)
            # -----------------------------------------
            ocr_result = smart_ocr_extraction(image_path, doc_type)

            if not isinstance(ocr_result, dict):
                ocr_result = {"raw_text": str(ocr_result)}

            # -----------------------------------------
            # STEP 2 — AUTO DETECT FROM OCR TEXT
            # -----------------------------------------
            detected_doc_type = doc_type

            if auto_detect and not doc_type:
                combined_text = ocr_result.get("raw_text", "")
                detected = analyzer.detect_document_type_from_text(combined_text)
                detected_doc_type = detected if detected != "unknown" else None

            # -----------------------------------------
            # STEP 3 — QUALITY WARNINGS
            # -----------------------------------------
            quality_warnings = []

            if is_blurry:
                quality_warnings.append("Document is blurry - accuracy may be reduced")

            if quality_scores.get("overall_score", 1) < 0.5:
                quality_warnings.append("Low overall image quality detected")

            if quality_scores.get("contrast_score", 1) < 0.3:
                quality_warnings.append("Low contrast detected")

            # -----------------------------------------
            # STEP 4 — METADATA
            # -----------------------------------------
            ocr_result["_metadata"] = {
                "page_number": idx,
                "quality_scores": quality_scores,
                "blur_detected": is_blurry,
                "blur_score": float(blur_score),
                "warnings": quality_warnings,
                "document_type": detected_doc_type or "unknown",
                "processing_method": "enhanced_ocr",
            }

            extracted_results[page_key] = ocr_result

        except Exception as e:
            logger.error(f"Error processing {page_key}: {e}")

            extracted_results[page_key] = {
                "error": f"Processing failed: {str(e)}",
                "_metadata": {
                    "page_number": idx,
                    "processing_method": "error",
                },
            }

    # -----------------------------------------
    # SAFE PDF CLEANUP
    # -----------------------------------------
    if ext == ".pdf" and image_paths != [file_path]:
        for path in image_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as cleanup_error:
                logger.warning(f"Cleanup failed for {path}: {cleanup_error}")

    return extracted_results


# -------------------------------------------------
# 🔹 BATCH PROCESSING
# -------------------------------------------------
def batch_process_documents(file_paths: List[str], doc_types: List[str] = None) -> Dict:
    # ... unchanged
    pass


# -------------------------------------------------
# 🔹 BACKWARD COMPATIBILITY
# -------------------------------------------------
def process_document_file(file_path: str, doc_type: str = None) -> Dict:
    # ... unchanged
    pass


# -------------------------------------------------
# 🔹 UTILITY FUNCTIONS
# -------------------------------------------------
def get_supported_document_types() -> List[str]:
    # ... unchanged
    pass


def validate_ocr_environment() -> Dict[str, bool]:
    # ... unchanged
    pass


def extract_text_from_document(file_path):
    """
    Extract text from image or PDF document.
    This function is kept for backward compatibility,
    but internally it uses the safe pipeline.
    """
    try:
        result = process_document_file_enhanced(file_path)
        return build_ocr_text(result)
    except Exception as e:
        logger.error(f"extract_text_from_document failed: {e}")
        return ""


def build_ocr_text(extracted_data: dict) -> str:
    # ... unchanged
    pass


# Keep OCR_HANDLERS definition if needed
if openbharatocr:
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
