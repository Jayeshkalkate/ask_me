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

# Set Tesseract command path
pytesseract.pytesseract.tesseract_cmd = getattr(
    settings, "PYTESSERACT_CMD", "tesseract"
)

OCR_HANDLERS = {}

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


def safe_read_image(image_path: str) -> np.ndarray:
    """
    Safely read image from disk.
    Raises clear error if OpenCV cannot load it.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"OpenCV failed to read image: {image_path}")

    return img


class OCRPreprocessor:
    """Advanced image preprocessing for OCR optimization"""

    @staticmethod
    def auto_rotate_image(image: np.ndarray) -> np.ndarray:
        """Auto-rotate image to correct orientation"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Use Tesseract to detect orientation
            osd = pytesseract.image_to_osd(gray)
            rotation = 0
            for line in osd.split("\n"):
                match = re.search(r"Rotate:\s+(\d+)", line)
                if match:
                    rotation = int(match.group(1))
                    break

            if rotation != 0:
                # Rotate the image
                if rotation == 90:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                elif rotation == 270:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        except Exception as e:
            logger.warning(f"Auto-rotation failed: {e}")

        return image

    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """Remove shadows using morphological operations"""
        rgb_planes = cv2.split(image)
        result_planes = []

        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )
            result_planes.append(norm_img)

        return cv2.merge(result_planes)

    @staticmethod
    def enhance_resolution(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Enhance image resolution using super-resolution or interpolation"""
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Use INTER_CUBIC for better quality
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Deskew the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.bitwise_not(gray)

        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find coordinates of all pixel values > 0
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return image

        # Get angle of rotation
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated


class DocumentAnalyzer:
    """Analyze document quality and characteristics"""

    @staticmethod
    def calculate_image_quality_score(image_path: str) -> Dict[str, float]:
        try:
            img = safe_read_image(image_path)
        except Exception:
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

    # ✅ MUST BE HERE (same indentation)
    @staticmethod
    def detect_document_type_from_text(text: str) -> str:
        if not text:
            return "unknown"

        text = text.lower()

        if "aadhaar" in text or "government of india" in text:
            return "aadhaar_front"
        if "income tax department" in text:
            return "pan"
        if "driving licence" in text:
            return "dl"
        if "passport" in text:
            return "passport"
        if "election commission" in text:
            return "voter_front"
        if "vehicle" in text and "registration" in text:
            return "rc"

        return "unknown"

# -------------------------------------------------
# 🔹 ENHANCED IMAGE QUALITY CHECK
# -------------------------------------------------
def is_image_blurry(image_path: str, threshold: float = 100.0) -> Tuple[bool, float]:
    """Enhanced blur detection with multiple metrics"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError()
    except:
        return True, 0.0

    # Method 1: Variance of Laplacian
    fm_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()

    # Method 2: FFT-based blur detection
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)

    # Calculate high frequency content
    rows, cols = img.shape
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


# -------------------------------------------------
# 🔹 ADVANCED IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image_advanced(
    image_path: str, enhancement_level: str = "auto"
) -> np.ndarray:
    """
    Advanced image preprocessing pipeline with multiple enhancement levels
    """
    img = safe_read_image(image_path)

    preprocessor = OCRPreprocessor()

    # Step 1: Auto-rotate
    img = preprocessor.auto_rotate_image(img)

    # Step 2: Remove shadows
    img = preprocessor.remove_shadows(img)

    # Step 3: Deskew
    img = preprocessor.deskew_image(img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: Denoise with advanced methods
    denoised = cv2.fastNlMeansDenoising(
        gray, h=15, templateWindowSize=7, searchWindowSize=21
    )

    # Step 5: Contrast enhancement based on enhancement level
    if enhancement_level == "aggressive":
        # CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Global contrast stretch
        enhanced = exposure.rescale_intensity(
            enhanced, in_range="image", out_range=(0, 255)
        )

    elif enhancement_level == "moderate":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

    else:  # auto or mild
        # Analyze image statistics to decide enhancement
        mean_intensity = np.mean(denoised)
        std_intensity = np.std(denoised)

        if std_intensity < 25:  # Low contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
        else:
            enhanced = denoised

    # 🔧 FIX: Ensure image is single-channel grayscale before adaptiveThreshold
    if len(enhanced.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # Step 6: Adaptive thresholding with multiple methods
    if enhancement_level == "aggressive":
        # Try multiple binarization methods
        thresh1 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10
        )
        thresh2 = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

        # Combine results (you can choose one or combine)
        final_thresh = cv2.bitwise_and(thresh1, thresh2)
    else:
        final_thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # Step 7: Morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(final_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # Step 8: Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(cleaned, -1, kernel)

    return sharpened


# -------------------------------------------------
# 🔹 ENHANCED PDF TO IMAGE CONVERSION
# -------------------------------------------------
def pdf_to_images_enhanced(pdf_path: str, dpi: int = 300, poppler_path: str = None):
    try:
        if poppler_path and os.path.exists(poppler_path):
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path, dpi=dpi)

        if not images:
            raise ValueError("PDF has no pages")

        temp_dir = tempfile.mkdtemp()
        img_paths = []

        for i, img in enumerate(images, 1):
            img_path = os.path.join(temp_dir, f"page_{i}.jpg")
            img.save(img_path, "JPEG", quality=95)
            img_paths.append(img_path)

        return img_paths

    except Exception as e:
        logger.error(f"PDF conversion failed: {e}")
        raise

# -------------------------------------------------
# 🔹 SMART OCR WITH FALLBACK STRATEGY
# -------------------------------------------------
def smart_ocr_extraction(
    image_path: str, doc_type: str = None, retry_count: int = 3
) -> Dict:
    """
    Smart OCR extraction with:
    - Multi-level image enhancement
    - Structured OCR (OpenBharatOCR) if available
    - Tesseract fallback with multiple configs
    - Clean failure detection (no fake text)
    """

    results = {}
    enhancement_levels = ["mild", "moderate", "aggressive"]

    for attempt in range(retry_count):
        temp_processed_path = None

        try:
            logger.info(f"OCR attempt {attempt + 1}/{retry_count}")

            # -----------------------------------------
            # STEP 1 — PREPROCESS IMAGE
            # -----------------------------------------
            enhancement = enhancement_levels[min(attempt, len(enhancement_levels) - 1)]
            processed_img = preprocess_image_advanced(image_path, enhancement)

            # -----------------------------------------
            # STEP 2 — SAVE TEMP IMAGE
            # -----------------------------------------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
                cv2.imwrite(temp_img.name, processed_img)
                temp_processed_path = temp_img.name

            # -----------------------------------------
            # STEP 3 — TRY STRUCTURED OCR (OpenBharatOCR)
            # -----------------------------------------
            if doc_type and doc_type in OCR_HANDLERS:
                try:
                    logger.info(f"Trying OpenBharatOCR for doc_type={doc_type}")

                    structured_data = OCR_HANDLERS[doc_type](temp_processed_path)

                    # Convert string JSON to dict
                    if isinstance(structured_data, str):
                        try:
                            structured_data = json.loads(structured_data)
                        except json.JSONDecodeError:
                            structured_data = {"raw_text": structured_data}

                    # Validate meaningful data
                    if isinstance(structured_data, dict) and structured_data:
                        meaningful = any(
                            v
                            and isinstance(v, str)
                            and re.search(r"[A-Za-z0-9]{4,}", v)
                            for v in structured_data.values()
                        )

                        if meaningful:
                            logger.info("Structured OCR successful")
                            results.update(structured_data)
                            break

                except Exception as e:
                    logger.warning(f"OpenBharatOCR failed: {e}")

            # -----------------------------------------
            # STEP 4 — FALLBACK TO TESSERACT OCR
            # -----------------------------------------
            tesseract_configs = [
                "--oem 3 --psm 6",  # block text
                "--oem 3 --psm 4",  # column text
                "--oem 3 --psm 11",  # sparse text
                "--oem 3 --psm 8",  # single word
                "--oem 3 --psm 13",  # raw line
            ]

            for config in tesseract_configs:
                try:
                    text = pytesseract.image_to_string(
                        processed_img, config=config
                    ).strip()

                    if len(text) > 10:
                        logger.info(f"Tesseract success with config: {config}")
                        results["raw_text"] = text
                        break

                except Exception as e:
                    logger.warning(f"Tesseract config failed ({config}): {e}")

            if results:
                break

        except Exception as e:
            logger.error(f"OCR attempt {attempt + 1} failed: {e}")

        finally:
            # -----------------------------------------
            # STEP 5 — CLEAN TEMP FILE
            # -----------------------------------------
            if temp_processed_path and os.path.exists(temp_processed_path):
                try:
                    os.remove(temp_processed_path)
                except Exception as e:
                    logger.warning(f"Temp file cleanup failed: {e}")

    # -----------------------------------------
    # STEP 6 — FINAL RESULT
    # -----------------------------------------
    if not results:
        logger.warning("OCR failed completely — no readable text")
        return {"status": "failed", "raw_text": ""}

    return {"status": "success", **results}


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
        except Exception as e:
            return {"error": f"PDF conversion failed: {str(e)}"}

    extracted_results = {}
    analyzer = DocumentAnalyzer()

    # -----------------------------------------
    # PROCESS EACH PAGE
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
            # STEP 1 — RUN OCR FIRST (ONLY ONCE)
            # -----------------------------------------
            ocr_result = smart_ocr_extraction(image_path, doc_type)

            if not isinstance(ocr_result, dict):
                ocr_result = {"raw_text": str(ocr_result)}

            # -----------------------------------------
            # STEP 2 — AUTO DETECT FROM OCR TEXT
            # -----------------------------------------
            detected_doc_type = doc_type

            if auto_detect and not doc_type:
                combined_text = ""

                if "raw_text" in ocr_result:
                    combined_text = ocr_result["raw_text"]
                else:
                    combined_text = " ".join(
                        str(v) for k, v in ocr_result.items() if k != "_metadata"
                    )

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
    """Process multiple documents in batch"""
    results = {}

    for i, file_path in enumerate(file_paths):
        doc_type = doc_types[i] if doc_types and i < len(doc_types) else None

        try:
            result = process_document_file_enhanced(file_path, doc_type)
            results[os.path.basename(file_path)] = result
        except Exception as e:
            results[os.path.basename(file_path)] = {"error": str(e)}

    return results


# -------------------------------------------------
# 🔹 BACKWARD COMPATIBILITY
# -------------------------------------------------
def process_document_file(file_path: str, doc_type: str = None) -> Dict:
    """Legacy function for backward compatibility"""
    return process_document_file_enhanced(file_path, doc_type, auto_detect=True)


# -------------------------------------------------
# 🔹 UTILITY FUNCTIONS
# -------------------------------------------------
def get_supported_document_types() -> List[str]:
    """Get list of supported document types"""
    return list(OCR_HANDLERS.keys())


def validate_ocr_environment() -> Dict[str, bool]:
    """Validate that all required OCR components are available"""

    checks = {
        "tesseract": False,
        "openbharatocr": False,
        "pdf2image": False,
        "opencv": False,
    }

    # ✅ Tesseract check
    try:
        pytesseract.get_tesseract_version()
        checks["tesseract"] = True
    except Exception:
        pass

    # ✅ OpenBharatOCR check (use global import result)
    if openbharatocr is not None:
        checks["openbharatocr"] = True

    # ✅ PDF2Image check
    try:
        convert_from_path
        checks["pdf2image"] = True
    except Exception:
        pass

    # ✅ OpenCV check
    try:
        cv2.__version__
        checks["opencv"] = True
    except Exception:
        pass

    return checks


def extract_text_from_document(file_path):
    """
    Extract text from image or PDF document.
    """

    text = ""

    try:
        # Handle Images
        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            import pytesseract
            from PIL import Image

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)

        # Handle PDF
        elif file_path.lower().endswith(".pdf"):
            from pdf2image import convert_from_path
            import pytesseract

            pages = convert_from_path(file_path)
            for page in pages:
                text += pytesseract.image_to_string(page)

        return text

    except Exception as e:
        print("OCR Error:", e)
        return ""

def build_ocr_text(extracted_data: dict) -> str:
    """
    Build combined OCR text from extracted pages.
    Safe and fast.
    """
    text_parts = []

    if not isinstance(extracted_data, dict):
        return ""

    for page_data in extracted_data.values():
        if not isinstance(page_data, dict):
            continue

        if "raw_text" in page_data:
            text_parts.append(page_data["raw_text"])
        else:
            for key, value in page_data.items():
                if key != "_metadata" and value:
                    text_parts.append(str(value))

    return " ".join(text_parts).strip()
