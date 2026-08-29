import os
import re
import logging
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# Try importing optional dependencies
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import openbharatocr
except ImportError:
    openbharatocr = None

from django.conf import settings

logger = logging.getLogger(__name__)
print("🔥 OCR_UTILS LOADED (Production Ready v3 - All Document Types)")

# ============================================================
#  CONFIGURATION
# ============================================================
TESSERACT_CONFIDENCE_THRESHOLD = 60
TESSERACT_PSM = 6
TESSERACT_OEM = 3
OCR_DPI = 300
PDF_QUALITY = 95
BLUR_THRESHOLD = 100.0
MIN_TEXT_LENGTH_FOR_OCR = 10
MAX_IMAGE_DIMENSION = 4000

# ============================================================
#  TESSERACT SETUP
# ============================================================
TESSERACT_AVAILABLE = False
TESSERACT_PATH = getattr(settings, "PYTESSERACT_CMD", None)

if pytesseract is not None:
    if TESSERACT_PATH and os.path.exists(TESSERACT_PATH):
        try:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            TESSERACT_AVAILABLE = True
            logger.info(f"✅ Tesseract configured: {TESSERACT_PATH}")
        except Exception as e:
            logger.warning(f"⚠ Failed to initialize Tesseract: {e}")
    else:
        logger.warning("⚠ Tesseract not found — pytesseract OCR disabled")
else:
    logger.warning("⚠ pytesseract not installed — OCR disabled")

# ============================================================
#  CUSTOM EXCEPTIONS
# ============================================================
class OCRException(Exception):
    pass

class ImageLoadError(OCRException):
    pass

class PDFConversionError(OCRException):
    pass

class UnsupportedDocumentType(OCRException):
    pass

# ============================================================
#  DOCUMENT TYPE MAPPING
# ============================================================
DOCUMENT_TYPE_MAPPING = {
    # Identity Documents
    "aadhaar_card": {
        "display_name": "Aadhaar Card",
        "openbharat_key": "aadhaar_card",
        "keywords": ["aadhaar", "uidai", "enrolment", "unique identification"],
        "regex_patterns": [r"\b\d{4}\s?\d{4}\s?\d{4}\b"],
    },
    "pan_card": {
        "display_name": "PAN Card",
        "openbharat_key": "pan",
        "keywords": ["pan", "permanent account number", "income tax"],
        "regex_patterns": [r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"],
    },
    "voter_id_card": {
        "display_name": "Voter ID Card",
        "openbharat_key": "voter_front",
        "keywords": ["voter", "epic", "election", "constituency"],
        "regex_patterns": [r"\b[A-Z]{3}[0-9]{7}\b"],
    },
    "passport": {
        "display_name": "Passport",
        "openbharat_key": "passport",
        "keywords": ["passport", "republic of india", "file number"],
        "regex_patterns": [r"\b[A-Z][0-9]{7}\b"],
    },
    "driving_license": {
        "display_name": "Driving License",
        "openbharat_key": "driving_license",
        "keywords": ["driving", "license", "licence", "transport"],
        "regex_patterns": [r"\b[A-Z]{2}\d{2}\s?\d{11}\b"],
    },
    "vehicle_registration_certificate": {
        "display_name": "Vehicle Registration Certificate",
        "openbharat_key": "rc",
        "keywords": ["registration", "vehicle", "rc", "chassis", "engine"],
        "regex_patterns": [r"\b[A-Z]{2}[0-9]{2}[A-Z]{4}[0-9]{4}\b"],
    },
    
    # Civil & Residency Documents
    "domicile_certificate": {
        "display_name": "Domicile Certificate",
        "openbharat_key": None,
        "keywords": ["domicile", "resident", "residence", "domicile certificate"],
        "regex_patterns": [],
    },
    "nationality_certificate": {
        "display_name": "Nationality Certificate",
        "openbharat_key": None,
        "keywords": ["nationality", "citizenship", "nationality certificate"],
        "regex_patterns": [],
    },
    "birth_certificate": {
        "display_name": "Birth Certificate",
        "openbharat_key": None,
        "keywords": ["birth", "born", "birth certificate", "date of birth"],
        "regex_patterns": [],
    },
    "marriage_certificate": {
        "display_name": "Marriage Certificate",
        "openbharat_key": None,
        "keywords": ["marriage", "married", "spouse", "marriage certificate"],
        "regex_patterns": [],
    },
    "death_certificate": {
        "display_name": "Death Certificate",
        "openbharat_key": None,
        "keywords": ["death", "died", "deceased", "death certificate"],
        "regex_patterns": [],
    },
    
    # Land & Property Documents
    "property_card": {
        "display_name": "Property Card",
        "openbharat_key": None,
        "keywords": ["property", "survey", "7/12", "8a", "plot", "khata"],
        "regex_patterns": [],
    },
    "income_certificate": {
        "display_name": "Income Certificate",
        "openbharat_key": None,
        "keywords": ["income", "income certificate", "annual income"],
        "regex_patterns": [],
    },
    "ration_card": {
        "display_name": "Ration Card",
        "openbharat_key": None,
        "keywords": ["ration", "ration card", "family", "household"],
        "regex_patterns": [],
    },
    
    # Educational Documents
    "school_leaving_certificate": {
        "display_name": "School Leaving Certificate",
        "openbharat_key": None,
        "keywords": ["school leaving", "transfer certificate", "tc", "lc"],
        "regex_patterns": [],
    },
    "ssc_marksheet": {
        "display_name": "SSC Marksheet",
        "openbharat_key": None,
        "keywords": ["ssc", "10th", "matric", "secondary school"],
        "regex_patterns": [],
    },
    "hsc_marksheet": {
        "display_name": "HSC Marksheet",
        "openbharat_key": None,
        "keywords": ["hsc", "12th", "intermediate", "higher secondary"],
        "regex_patterns": [],
    },
    "degree_certificate": {
        "display_name": "Degree Certificate",
        "openbharat_key": None,
        "keywords": ["degree", "bachelor", "master", "university"],
        "regex_patterns": [],
    },
    "board_passing_certificate": {
        "display_name": "Board Passing Certificate",
        "openbharat_key": None,
        "keywords": ["board", "passing", "certificate", "board examination"],
        "regex_patterns": [],
    },
    
    # Category & Welfare Certificates
    "caste_certificate": {
        "display_name": "Caste Certificate",
        "openbharat_key": None,
        "keywords": ["caste", "category", "caste certificate", "obc", "sc", "st"],
        "regex_patterns": [],
    },
    "caste_validity_certificate": {
        "display_name": "Caste Validity Certificate",
        "openbharat_key": None,
        "keywords": ["caste validity", "validity certificate", "scrutiny"],
        "regex_patterns": [],
    },
    "non_creamy_layer_certificate": {
        "display_name": "Non-Creamy Layer Certificate",
        "openbharat_key": None,
        "keywords": ["non creamy", "ncl", "creamy layer", "obc"],
        "regex_patterns": [],
    },
    "ews_certificate": {
        "display_name": "EWS Certificate",
        "openbharat_key": None,
        "keywords": ["ews", "economically weaker", "financial"],
        "regex_patterns": [],
    },
    
    # Other Documents
    "gst_certificate": {
        "display_name": "GST Certificate",
        "openbharat_key": None,
        "keywords": ["gst", "goods and services tax", "gstin"],
        "regex_patterns": [r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\dZ\d\b"],
    },
    "invoice": {
        "display_name": "Invoice",
        "openbharat_key": None,
        "keywords": ["invoice", "bill", "tax invoice", "purchase"],
        "regex_patterns": [],
    },
    "other_document": {
        "display_name": "Other Document",
        "openbharat_key": None,
        "keywords": [],
        "regex_patterns": [],
    },
}

# ============================================================
#  IMAGE LOADER
# ============================================================
def load_image_strict(image_path: str) -> np.ndarray:
    """Load image with validation; resize if too large."""
    if cv2 is None:
        raise OCRException("OpenCV (cv2) is not installed")
    if not os.path.exists(image_path):
        raise ImageLoadError(f"Image file not found: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ImageLoadError(f"OpenCV failed to read image (corrupt or unsupported): {image_path}")
    if img.size == 0:
        raise ImageLoadError(f"Image has zero pixels: {image_path}")

    h, w = img.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    return img

# ============================================================
#  SAFE TESSERACT WRAPPER
# ============================================================
def run_tesseract_safe(
    image: np.ndarray,
    config: str = f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}",
    lang: str = "eng+hin",
    confidence_threshold: int = TESSERACT_CONFIDENCE_THRESHOLD,
) -> str:
    """Run Tesseract OCR with safety checks."""
    if not TESSERACT_AVAILABLE:
        logger.warning("⚠ Tesseract not available — OCR skipped")
        return ""
    if image is None or image.size == 0:
        logger.warning("Empty image passed to Tesseract")
        return ""
    try:
        data = pytesseract.image_to_data(
            image,
            config=config,
            lang=lang,
            output_type=pytesseract.Output.DICT
        )
        filtered = []
        for text, conf in zip(data["text"], data["conf"]):
            try:
                conf = float(conf)
            except (ValueError, TypeError):
                continue
            text = text.strip()
            if conf >= confidence_threshold and text:
                filtered.append(text)
        return " ".join(filtered)
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}", exc_info=True)
        return ""

# ============================================================
#  OCR TEXT CLEANING
# ============================================================
def clean_ocr_text(text: str) -> str:
    """Clean OCR text by removing noise and normalizing whitespace."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s/\-:,\.\(\)\']", "", text)
    return text.strip()

# ============================================================
#  IMAGE PREPROCESSORS
# ============================================================
class OCRPreprocessor:
    """Image preprocessing utilities for better OCR results."""
    
    @staticmethod
    def remove_shadows(image: np.ndarray) -> np.ndarray:
        """Remove shadows from image."""
        if cv2 is None:
            return image
        bg = cv2.medianBlur(image, 21)
        diff = cv2.absdiff(image, bg)
        return cv2.convertScaleAbs(diff, alpha=1.5, beta=20)

    @staticmethod
    def enhance_resolution(image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """Enhance image resolution."""
        if cv2 is None or scale_factor <= 1.0:
            return image
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Deskew rotated image."""
        if cv2 is None:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is not None:
            angle = 0.0
            for rho, theta in lines[:, 0]:
                if theta != 0:
                    angle = (theta * 180 / np.pi) - 90
                    break
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        return image

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction."""
        if cv2 is None:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        if cv2 is None:
            return image
        return cv2.medianBlur(image, 3)

    @staticmethod
    def auto_rotate(image: np.ndarray) -> np.ndarray:
        """Auto-rotate image based on text orientation."""
        if cv2 is None or np is None:
            return image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detect text orientation
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 2:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))
        return image

# ============================================================
#  DOCUMENT ANALYZER
# ============================================================
class DocumentAnalyzer:
    """Analyze document quality and type."""
    
    @staticmethod
    def calculate_image_quality_score(image_path: str) -> Dict[str, float]:
        """Calculate image quality metrics."""
        if cv2 is None or np is None:
            return {"overall_score": 0.0, "blur_score": 0.0, "contrast_score": 0.0, "brightness_score": 0.0}
        try:
            img = load_image_strict(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except ImageLoadError:
            return {"overall_score": 0.0, "blur_score": 0.0, "contrast_score": 0.0, "brightness_score": 0.0}
        
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        blur_norm = min(blur / 1000, 1.0)
        contrast_norm = min(contrast / 64, 1.0)
        brightness_norm = 1 - abs(brightness - 127) / 127
        brightness_norm = max(0.0, min(brightness_norm, 1.0))
        overall = (blur_norm + contrast_norm + brightness_norm) / 3.0
        
        return {
            "overall_score": float(overall),
            "blur_score": float(blur_norm),
            "contrast_score": float(contrast_norm),
            "brightness_score": float(brightness_norm),
        }

    @staticmethod
    def detect_document_type_from_text(text: str) -> Tuple[str, float]:
        """
        Detect document type from OCR text with confidence score.
        Returns (doc_type, confidence).
        """
        if not text:
            return "unknown", 0.0
        
        text_lower = text.lower()
        confidence = 0.0
        detected_type = "unknown"
        
        # Check each document type
        for doc_type, info in DOCUMENT_TYPE_MAPPING.items():
            if not info["keywords"]:
                continue
            
            # Count matching keywords
            matches = sum(1 for keyword in info["keywords"] if keyword in text_lower)
            if matches > 0:
                # Calculate confidence based on matches
                type_confidence = matches / len(info["keywords"])
                if type_confidence > confidence:
                    confidence = type_confidence
                    detected_type = doc_type
        
        # Check for regex patterns if keywords didn't match strongly
        if confidence < 0.3:
            for doc_type, info in DOCUMENT_TYPE_MAPPING.items():
                for pattern in info["regex_patterns"]:
                    if re.search(pattern, text):
                        confidence = 0.6
                        detected_type = doc_type
                        break
        
        return detected_type, confidence

# ============================================================
#  BLUR DETECTION
# ============================================================
def is_image_blurry(image_path: str, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """Check if image is blurry using Laplacian variance."""
    if cv2 is None or np is None:
        return True, 0.0
    try:
        img = load_image_strict(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except ImageLoadError:
        return True, 0.0
    
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    fft_shift[crow - 30: crow + 30, ccol - 30: ccol + 30] = 0
    fft_ishift = np.fft.ifftshift(fft_shift)
    img_back = np.fft.ifft2(fft_ishift)
    img_back = np.abs(img_back)
    high_freq = np.mean(img_back)
    score = (lap_var + high_freq) / 2.0
    return score < threshold, score

# ============================================================
#  PDF TO IMAGES
# ============================================================
def pdf_to_images_enhanced(pdf_path: str, dpi: int = OCR_DPI, poppler_path: Optional[str] = None) -> List[str]:
    """Convert PDF to images with enhanced quality."""
    if convert_from_path is None:
        raise PDFConversionError("pdf2image is not installed. Cannot convert PDF.")
    if not os.path.exists(pdf_path):
        raise PDFConversionError(f"PDF file not found: {pdf_path}")
    
    try:
        if poppler_path and os.path.exists(poppler_path):
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path, dpi=dpi)
    except Exception as e:
        raise PDFConversionError(f"Failed to convert PDF: {e}") from e
    
    if not images:
        raise PDFConversionError("PDF has no pages or conversion produced empty result")
    
    temp_dir = tempfile.mkdtemp()
    img_paths = []
    for i, img in enumerate(images, start=1):
        img_path = os.path.join(temp_dir, f"page_{i}.jpg")
        img.save(img_path, "JPEG", quality=PDF_QUALITY)
        img_paths.append(img_path)
    return img_paths

# ============================================================
#  SINGLE IMAGE OCR
# ============================================================
def extract_text_with_tesseract(image_path: str, lang: str = "eng+hin") -> str:
    """Extract text from a single image using Tesseract."""
    if cv2 is None:
        logger.error("OpenCV not installed, cannot perform OCR")
        return ""
    
    try:
        img = load_image_strict(image_path)
        
        # Preprocess image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if thresh is None or thresh.size == 0:
            logger.error("Threshold image invalid before OCR")
            return ""
        
        if thresh.shape[0] < 10 or thresh.shape[1] < 10:
            logger.warning("Threshold image too small for OCR")
        
        text = run_tesseract_safe(
            thresh,
            config=f"--oem {TESSERACT_OEM} --psm {TESSERACT_PSM}",
            lang=lang,
            confidence_threshold=TESSERACT_CONFIDENCE_THRESHOLD
        )
        return clean_ocr_text(text)
    except ImageLoadError as e:
        logger.error(f"Image load failed: {e}")
        return ""
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return ""

# ============================================================
#  SMART OCR EXTRACTION
# ============================================================
def smart_ocr_extraction(image_path: str, doc_type: Optional[str] = None) -> Dict:
    """Extract text with smart preprocessing based on document type."""
    try:
        logger.info(f"Starting OCR on {image_path}")
        
        # Load image
        img = load_image_strict(image_path)
        
        # Apply preprocessing based on document type
        preprocessor = OCRPreprocessor()
        
        # Basic preprocessing
        img = preprocessor.deskew_image(img)
        img = preprocessor.remove_shadows(img)
        img = preprocessor.remove_noise(img)
        img = preprocessor.enhance_resolution(img)
        
        # If document is a specific type, apply targeted preprocessing
        if doc_type:
            if doc_type in ["aadhaar_card", "pan_card", "voter_id_card"]:
                # For ID cards, apply adaptive threshold
                img = preprocessor.apply_adaptive_threshold(img)
            
            if doc_type in ["passport", "driving_license"]:
                # For documents with photos, enhance contrast
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                img = cv2.equalizeHist(gray)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Save preprocessed image temporarily
        temp_img_path = image_path + "_preprocessed.jpg"
        cv2.imwrite(temp_img_path, img)
        
        # Extract text
        raw_text = extract_text_with_tesseract(temp_img_path)
        
        # Cleanup temp file
        try:
            os.unlink(temp_img_path)
        except OSError:
            pass
        
        if not raw_text or len(raw_text.strip()) < MIN_TEXT_LENGTH_FOR_OCR:
            logger.warning("OCR returned very little text")
            return {"status": "failed", "raw_text": raw_text or ""}
        
        # Try OpenBharatOCR if available
        structured_result = None
        if doc_type and openbharatocr:
            handler_key = DOCUMENT_TYPE_MAPPING.get(doc_type, {}).get("openbharat_key")
            if handler_key and handler_key in OCR_HANDLERS:
                try:
                    logger.info(f"Running OpenBharatOCR for {doc_type}")
                    structured_result = OCR_HANDLERS[handler_key](image_path)
                except Exception as e:
                    logger.warning(f"OpenBharatOCR failed for {doc_type}: {e}", exc_info=True)
        
        return {
            "status": "success",
            "raw_text": raw_text,
            "structured_data": structured_result,
        }
    except Exception as e:
        logger.error(f"Smart OCR extraction failed: {e}", exc_info=True)
        return {"status": "failed", "raw_text": ""}

# ============================================================
#  MAIN DOCUMENT PROCESSING
# ============================================================
def process_document_file_enhanced(
    file_path: str,
    doc_type: Optional[str] = None,
    auto_detect: bool = True,
    lang: str = "eng+hin"
) -> Dict:
    """
    Process a document file with enhanced OCR.
    Supports all document types.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    image_paths = []
    
    # Convert PDF to images
    if ext == ".pdf":
        try:
            image_paths = pdf_to_images_enhanced(file_path)
        except PDFConversionError as e:
            return {"error": f"PDF conversion failed: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected PDF error: {str(e)}"}
    else:
        image_paths = [file_path]

    results = {}
    analyzer = DocumentAnalyzer()
    detected_doc_type = doc_type
    detection_confidence = 0.0
    
    try:
        for idx, img_path in enumerate(image_paths, start=1):
            page_key = f"page_{idx}"
            try:
                # Quality analysis
                quality_scores = analyzer.calculate_image_quality_score(img_path)
                is_blurry, blur_score = is_image_blurry(img_path)
                
                # OCR extraction
                ocr_result = smart_ocr_extraction(img_path, doc_type)
                if not isinstance(ocr_result, dict):
                    ocr_result = {"raw_text": str(ocr_result)}
                
                # Auto-detect document type
                if auto_detect and not doc_type:
                    detected, confidence = analyzer.detect_document_type_from_text(
                        ocr_result.get("raw_text", "")
                    )
                    if detected != "unknown" and confidence > detection_confidence:
                        detected_doc_type = detected
                        detection_confidence = confidence
                
                # Build warnings
                warnings = []
                if is_blurry:
                    warnings.append("Document is blurry - accuracy may be reduced")
                if quality_scores.get("overall_score", 1) < 0.5:
                    warnings.append("Low overall image quality detected")
                if quality_scores.get("contrast_score", 1) < 0.3:
                    warnings.append("Low contrast detected")
                
                # Add metadata
                ocr_result["_metadata"] = {
                    "page_number": idx,
                    "total_pages": len(image_paths),
                    "quality_scores": quality_scores,
                    "blur_detected": is_blurry,
                    "blur_score": float(blur_score),
                    "warnings": warnings,
                    "document_type": detected_doc_type or "unknown",
                    "detection_confidence": detection_confidence,
                    "processing_method": "enhanced_ocr_v3",
                    "tesseract_available": TESSERACT_AVAILABLE,
                }
                results[page_key] = ocr_result
                
            except Exception as e:
                logger.error(f"Error processing page {idx}: {e}", exc_info=True)
                results[page_key] = {
                    "error": f"Processing failed: {str(e)}",
                    "_metadata": {
                        "page_number": idx,
                        "total_pages": len(image_paths),
                        "processing_method": "error",
                    }
                }

    finally:
        # Cleanup temp PDF images
        if ext == ".pdf" and image_paths:
            temp_dir = os.path.dirname(image_paths[0]) if image_paths else None
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as cleanup_error:
                    logger.warning(f"Cleanup failed for temp PDF images: {cleanup_error}")

    # Add overall document summary
    results["_summary"] = {
        "total_pages": len(image_paths),
        "successful_pages": sum(1 for p in results.values() if isinstance(p, dict) and "error" not in p),
        "failed_pages": sum(1 for p in results.values() if isinstance(p, dict) and "error" in p),
        "document_type": detected_doc_type,
        "detection_confidence": detection_confidence,
        "quality_score": np.mean([p.get("_metadata", {}).get("quality_scores", {}).get("overall_score", 0) 
                                  for p in results.values() if isinstance(p, dict)]) if np else 0,
    }
    
    return results

# ============================================================
#  BATCH PROCESSING
# ============================================================
def batch_process_documents(
    file_paths: List[str],
    doc_types: Optional[List[str]] = None,
    lang: str = "eng+hin"
) -> Dict[str, Dict]:
    """Process multiple documents in batch."""
    if doc_types is None:
        doc_types = [None] * len(file_paths)
    else:
        doc_types = doc_types[:len(file_paths)]
    
    results = {}
    for path, d_type in zip(file_paths, doc_types):
        try:
            results[path] = process_document_file_enhanced(path, d_type, auto_detect=True, lang=lang)
        except Exception as e:
            results[path] = {"error": str(e)}
    return results

# ============================================================
#  BACKWARD COMPATIBILITY
# ============================================================
def process_document_file(file_path: str, doc_type: Optional[str] = None) -> Dict:
    """Backward compatibility wrapper."""
    return process_document_file_enhanced(file_path, doc_type, auto_detect=True)

# ============================================================
#  UTILITY FUNCTIONS
# ============================================================
def get_supported_document_types() -> List[str]:
    """Get list of all supported document types."""
    return [info["display_name"] for info in DOCUMENT_TYPE_MAPPING.values()]


def get_document_type_key(display_name: str) -> Optional[str]:
    """Get the document type key from display name."""
    for key, info in DOCUMENT_TYPE_MAPPING.items():
        if info["display_name"] == display_name:
            return key
    return None


def validate_ocr_environment() -> Dict[str, bool]:
    """Validate the OCR environment."""
    return {
        "tesseract_available": TESSERACT_AVAILABLE,
        "opencv_available": cv2 is not None,
        "numpy_available": np is not None,
        "pdf2image_available": convert_from_path is not None,
        "openbharatocr_available": openbharatocr is not None,
        "total_supported_docs": len(DOCUMENT_TYPE_MAPPING),
    }


def extract_text_from_document(file_path: str) -> str:
    """Extract text from a document file."""
    try:
        result = process_document_file_enhanced(file_path)
        return build_ocr_text(result)
    except Exception as e:
        logger.error(f"extract_text_from_document failed: {e}")
        return ""


def build_ocr_text(extracted_data: Dict) -> str:
    """Build a concatenated text from OCR results."""
    if not extracted_data:
        return ""
    text_parts = []
    for key, page_data in extracted_data.items():
        if key.startswith("_") or not isinstance(page_data, dict):
            continue
        if "raw_text" in page_data:
            text_parts.append(page_data["raw_text"])
        else:
            for field_key, field_value in page_data.items():
                if field_key != "_metadata" and field_value:
                    text_parts.append(f"{field_key}: {field_value}")
    return " ".join(text_parts)


def get_document_summary(ocr_result: Dict) -> Dict:
    """Get a summary of the OCR results."""
    if not ocr_result:
        return {}
    
    summary = {
        "pages_processed": 0,
        "pages_with_errors": 0,
        "document_type": "unknown",
        "quality_score": 0,
        "has_warnings": False,
        "warnings": [],
        "total_text_length": 0,
    }
    
    for key, page_data in ocr_result.items():
        if key.startswith("_") or not isinstance(page_data, dict):
            continue
        
        summary["pages_processed"] += 1
        metadata = page_data.get("_metadata", {})
        
        if "error" in page_data:
            summary["pages_with_errors"] += 1
        
        if metadata.get("warnings"):
            summary["has_warnings"] = True
            summary["warnings"].extend(metadata["warnings"])
        
        if metadata.get("document_type") and metadata["document_type"] != "unknown":
            summary["document_type"] = metadata["document_type"]
        
        if metadata.get("quality_scores", {}).get("overall_score", 0) > summary["quality_score"]:
            summary["quality_score"] = metadata["quality_scores"]["overall_score"]
        
        if page_data.get("raw_text"):
            summary["total_text_length"] += len(page_data["raw_text"])
    
    return summary


# ============================================================
#  SUPPORTED DOCUMENT TYPES DETAILS
# ============================================================
def get_document_type_info(doc_type: str) -> Optional[Dict]:
    """Get information about a specific document type."""
    return DOCUMENT_TYPE_MAPPING.get(doc_type)


def get_all_document_types_info() -> Dict:
    """Get information about all document types."""
    return DOCUMENT_TYPE_MAPPING


def get_document_types_by_category() -> Dict[str, List[str]]:
    """Get document types grouped by category."""
    categories = {
        "Identity Documents": ["aadhaar_card", "pan_card", "voter_id_card", "passport", 
                              "driving_license", "vehicle_registration_certificate"],
        "Civil & Residency": ["domicile_certificate", "nationality_certificate", 
                             "birth_certificate", "marriage_certificate", "death_certificate"],
        "Land & Property": ["property_card", "income_certificate", "ration_card"],
        "Educational": ["school_leaving_certificate", "ssc_marksheet", "hsc_marksheet", 
                       "degree_certificate", "board_passing_certificate"],
        "Category & Welfare": ["caste_certificate", "caste_validity_certificate", 
                              "non_creamy_layer_certificate", "ews_certificate"],
        "Other": ["gst_certificate", "invoice", "other_document"],
    }
    return categories


# ============================================================
#  OpenBharatOCR HANDLERS
# ============================================================
OCR_HANDLERS = {}
if openbharatocr:
    OCR_HANDLERS = {
        "pan": openbharatocr.pan,
        "aadhaar_card": openbharatocr.front_aadhaar,
        "aadhaar_front": openbharatocr.front_aadhaar,
        "aadhaar_back": openbharatocr.back_aadhaar,
        "driving_license": openbharatocr.driving_licence,
        "dl": openbharatocr.driving_licence,
        "passport": openbharatocr.passport,
        "voter_front": openbharatocr.voter_id_front,
        "voter_back": openbharatocr.voter_id_back,
        "rc": openbharatocr.vehicle_registration,
        "vehicle_registration": openbharatocr.vehicle_registration,
    }

print(f"✅ OCR_UTILS loaded with support for {len(DOCUMENT_TYPE_MAPPING)} document types")
print(f"📋 Document types: {', '.join(get_supported_document_types())}")