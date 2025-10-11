# C:\chatbot\ask_me\core\ocr_utils.py
import os
import cv2
import pytesseract
import numpy as np
from django.conf import settings
from pdf2image import convert_from_path
import openbharatocr
import json
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Dict, List, Optional, Union, Tuple
import re
import tempfile
from skimage import exposure, filters, restoration
import imutils

# Configure logging
logger = logging.getLogger(__name__)

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
                if "Rotate" in line:
                    rotation = int(line.split(":")[1].strip())
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find coordinates of all pixel values > 0
        coords = np.column_stack(np.where(thresh > 0))

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
        """Calculate multiple image quality metrics"""
        img = cv2.imread(image_path)
        if img is None:
            return {
                "overall_score": 0,
                "blur_score": 0,
                "contrast_score": 0,
                "brightness_score": 0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur score (variance of Laplacian)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast score (standard deviation)
        contrast_score = np.std(gray)

        # Brightness score (mean intensity)
        brightness_score = np.mean(gray)

        # Normalize scores
        blur_normalized = min(blur_score / 1000, 1.0)  # Assuming 1000 is good
        contrast_normalized = min(contrast_score / 64, 1.0)  # Assuming 64 is good
        brightness_normalized = 1 - abs(brightness_score - 127) / 127  # Ideal is 127

        overall_score = (
            blur_normalized + contrast_normalized + brightness_normalized
        ) / 3

        return {
            "overall_score": overall_score,
            "blur_score": blur_normalized,
            "contrast_score": contrast_normalized,
            "brightness_score": brightness_normalized,
        }

    @staticmethod
    def detect_document_type(image_path: str) -> str:
        """Auto-detect document type based on visual features"""
        # This is a simplified version - you can expand with ML models
        img = cv2.imread(image_path)
        if img is None:
            return "unknown"

        # Basic dimension-based detection
        height, width = img.shape[:2]
        aspect_ratio = width / height

        if 0.6 < aspect_ratio < 0.7:
            return "aadhaar_front"
        elif 1.3 < aspect_ratio < 1.5:
            return "dl"
        elif 0.7 < aspect_ratio < 0.8:
            return "pan"
        else:
            return "unknown"


# -------------------------------------------------
# 🔹 ENHANCED IMAGE QUALITY CHECK
# -------------------------------------------------
def is_image_blurry(image_path: str, threshold: float = 100.0) -> Tuple[bool, float]:
    """Enhanced blur detection with multiple metrics"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
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
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image file: {image_path}")

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
        gray, h=30, templateWindowSize=7, searchWindowSize=21
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
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
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
def pdf_to_images_enhanced(
    pdf_path: str, dpi: int = 300, poppler_path: str = None
) -> List[str]:
    """Enhanced PDF to image conversion with better quality control"""
    try:
        if poppler_path and os.path.exists(poppler_path):
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path, dpi=dpi)

        if not images:
            raise ValueError("PDF has no pages or could not be converted")

        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        img_paths = []

        for i, img in enumerate(images, start=1):
            # Enhance image before saving
            enhancer = ImageEnhance.Sharpness(img)
            img_enhanced = enhancer.enhance(1.5)

            img_filename = f"pdf_page_{i}_{os.path.basename(pdf_path)}.jpg"
            img_path = os.path.join(temp_dir, img_filename)

            # Save with high quality
            img_enhanced.save(img_path, "JPEG", quality=95, optimize=True)
            img_paths.append(img_path)

        return img_paths

    except Exception as e:
        logger.error(f"PDF to image conversion failed: {e}")
        raise


# -------------------------------------------------
# 🔹 SMART OCR WITH FALLBACK STRATEGY
# -------------------------------------------------
def smart_ocr_extraction(
    image_path: str, doc_type: str = None, retry_count: int = 3
) -> Dict:
    """
    Smart OCR extraction with multiple fallback strategies
    """
    results = {}
    enhancement_levels = ["mild", "moderate", "aggressive"]

    for attempt in range(retry_count):
        try:
            enhancement = enhancement_levels[min(attempt, len(enhancement_levels) - 1)]

            # Preprocess with current enhancement level
            processed_img = preprocess_image_advanced(image_path, enhancement)

            # Save processed image temporarily for OpenBharatOCR
            temp_processed_path = f"/tmp/processed_{os.path.basename(image_path)}"
            cv2.imwrite(temp_processed_path, processed_img)

            # Try OpenBharatOCR first if doc_type is specified
            if doc_type and doc_type in OCR_HANDLERS:
                try:
                    structured_data = OCR_HANDLERS[doc_type](temp_processed_path)

                    if isinstance(structured_data, str):
                        try:
                            structured_data = json.loads(structured_data)
                        except json.JSONDecodeError:
                            structured_data = {"raw_text": structured_data}

                    if isinstance(structured_data, dict) and structured_data:
                        # Validate that we got meaningful data
                        if any(
                            len(str(v).strip()) > 3
                            for v in structured_data.values()
                            if v and str(v).strip()
                        ):
                            results.update(structured_data)
                            logger.info(
                                f"Successfully extracted data using OpenBharatOCR (attempt {attempt + 1})"
                            )
                            break
                except Exception as e:
                    logger.warning(f"OpenBharatOCR attempt {attempt + 1} failed: {e}")

            # Fallback to Tesseract with different configurations
            tesseract_configs = [
                "--oem 3 --psm 6",  # Uniform block of text
                "--oem 3 --psm 4",  # Single column of text
                "--oem 3 --psm 8",  # Single word
                "--oem 3 --psm 13",  # Raw line
            ]

            for config in tesseract_configs:
                try:
                    text = pytesseract.image_to_string(
                        processed_img, config=config
                    ).strip()
                    if len(text) > 10:  # Meaningful text extracted
                        results["raw_text"] = text
                        logger.info(
                            f"Successfully extracted text using Tesseract (config: {config})"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Tesseract config {config} failed: {e}")

            if results:
                break

        except Exception as e:
            logger.error(f"OCR attempt {attempt + 1} failed: {e}")

    # Clean up temporary file
    try:
        if "temp_processed_path" in locals():
            os.remove(temp_processed_path)
    except:
        pass

    return results if results else {"raw_text": "No readable text could be extracted"}


# -------------------------------------------------
# 🔹 ENHANCED MAIN OCR PIPELINE
# -------------------------------------------------
def process_document_file_enhanced(
    file_path: str, doc_type: str = None, auto_detect: bool = True
) -> Dict:
    """
    Enhanced document processing with automatic quality assessment and optimization
    """
    # Validate file exists
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    image_paths = [file_path]

    # Convert PDF to images if needed
    if ext == ".pdf":
        try:
            image_paths = pdf_to_images_enhanced(file_path)
        except Exception as e:
            return {"error": f"PDF conversion failed: {str(e)}"}

    extracted_results = {}
    analyzer = DocumentAnalyzer()

    for idx, image_path in enumerate(image_paths, start=1):
        page_key = f"page_{idx}"

        try:
            # ---- Comprehensive Quality Assessment ----
            quality_scores = analyzer.calculate_image_quality_score(image_path)
            is_blurry, blur_score = is_image_blurry(image_path)

            # Auto-detect document type if not provided
            if auto_detect and not doc_type:
                detected_type = analyzer.detect_document_type(image_path)
                doc_type = detected_type if detected_type != "unknown" else None

            # ---- Quality Warnings ----
            quality_warnings = []
            if is_blurry:
                quality_warnings.append("Document is blurry - accuracy may be reduced")
            if quality_scores["overall_score"] < 0.5:
                quality_warnings.append("Low overall image quality detected")
            if quality_scores["contrast_score"] < 0.3:
                quality_warnings.append("Low contrast detected")

            # ---- Smart OCR Extraction ----
            ocr_result = smart_ocr_extraction(image_path, doc_type)

            # Add metadata
            ocr_result["_metadata"] = {
                "page_number": idx,
                "quality_scores": quality_scores,
                "blur_detected": is_blurry,
                "blur_score": float(blur_score),
                "warnings": quality_warnings,
                "document_type": doc_type or "unknown",
                "processing_method": "enhanced_ocr",
            }

            # Clean up redundant field
            if "raw_text" in ocr_result and len(ocr_result) > 2:
                # Only remove raw_text if we have structured data
                structured_fields = [
                    k for k in ocr_result.keys() if k not in ["raw_text", "_metadata"]
                ]
                if structured_fields:
                    del ocr_result["raw_text"]

            extracted_results[page_key] = ocr_result

        except Exception as e:
            logger.error(f"Error processing {page_key}: {e}")
            extracted_results[page_key] = {
                "error": f"Processing failed: {str(e)}",
                "_metadata": {"page_number": idx, "processing_method": "error"},
            }

    # Clean up temporary files for PDFs
    if ext == ".pdf" and image_paths != [file_path]:
        for path in image_paths:
            try:
                if path != file_path:  # Don't delete original
                    os.remove(path)
                    os.rmdir(os.path.dirname(path))
            except:
                pass

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

    try:
        pytesseract.get_tesseract_version()
        checks["tesseract"] = True
    except:
        pass

    try:
        import openbharatocr

        checks["openbharatocr"] = True
    except:
        pass

    try:
        from pdf2image import convert_from_path

        checks["pdf2image"] = True
    except:
        pass

    try:
        import cv2

        checks["opencv"] = True
    except:
        pass

    return checks


# Example usage and test function
if __name__ == "__main__":
    # Test the OCR environment
    env_status = validate_ocr_environment()
    print("Environment Status:", env_status)

    # Example usage
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        doc_type = sys.argv[2] if len(sys.argv) > 2 else None

        result = process_document_file_enhanced(file_path, doc_type)
        print(json.dumps(result, indent=2, ensure_ascii=False))
