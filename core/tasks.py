# core/tasks.py - Updated to use utils instead of views
# Celery is temporarily disabled - using threading fallback

import logging
from django.utils import timezone
from .models import Document, convert_numpy
from .ai_utils import detect_document_type
from .ocr_utils import process_document_file_enhanced
from .utils import clean_extracted_data, get_structured_fields_from_text, INTERNAL_KEYS

logger = logging.getLogger(__name__)

# Celery is temporarily disabled
# from celery import shared_task

# @shared_task
def process_document_in_background(doc_id: int) -> None:
    """Background worker for OCR processing."""
    try:
        doc = Document.objects.get(id=doc_id)
        if doc.processed and doc.error_message is None:
            return

        ocr_result = process_document_file_enhanced(doc.file.path)
        if not ocr_result or "error" in ocr_result:
            doc.processed = False
            doc.error_message = ocr_result.get("error", "OCR failed – no text detected")
            doc.extracted_data = {}
            doc.save(update_fields=["processed", "error_message", "extracted_data"])
            return

        # Build raw OCR text from all pages – skip internal keys (starting with "_")
        ocr_parts = []
        for page_key, page_data in ocr_result.items():
            if page_key.startswith("_") or not isinstance(page_data, dict):
                continue
            if "raw_text" in page_data:
                ocr_parts.append(page_data["raw_text"])
            else:
                for field, value in page_data.items():
                    if field not in INTERNAL_KEYS and value:
                        ocr_parts.append(str(value))
        ocr_text = " ".join(ocr_parts).strip()
        doc.extracted_text = ocr_text

        # Auto‑detect doc type
        if len(ocr_text) >= 20:
            doc.doc_type = detect_document_type(ocr_text) or "other_document"

        # Extract structured fields (rule‑based)
        structured_data = get_structured_fields_from_text(ocr_text)

        if structured_data:
            final_data = {"page_1": structured_data}
        else:
            cleaned = clean_extracted_data(ocr_result)
            if cleaned:
                first_page = next(iter(cleaned.values())) if isinstance(cleaned, dict) else cleaned
                if first_page and isinstance(first_page, dict):
                    final_data = {"page_1": first_page}
                else:
                    final_data = {"page_1": {"Content": ocr_text[:500]}}
            else:
                final_data = {"page_1": {"Content": ocr_text[:500]}}

        doc.extracted_data = convert_numpy(final_data)
        doc.processed = True
        doc.error_message = None
        doc.processed_at = timezone.now()
        doc.save()

        logger.info(f"Background processing completed for doc {doc_id}")

    except Document.DoesNotExist:
        logger.warning(f"Document {doc_id} not found during background processing")
    except Exception as e:
        logger.exception(f"Background OCR failed for doc {doc_id}")
        try:
            Document.objects.filter(id=doc_id).update(
                processed=False,
                error_message=str(e)[:500],
                extracted_data={},
            )
        except Exception as update_error:
            logger.error(f"Failed to update document {doc_id} after error: {update_error}")