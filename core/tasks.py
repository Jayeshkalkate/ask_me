# core/tasks.py - Background OCR processing (thread fallback)
import logging
import os
from django.utils import timezone
from .models import Document, convert_numpy
from .ai_utils import detect_document_type
from .ai_extract import extract_document_ai
from .utils import build_document_text_and_fields

logger = logging.getLogger(__name__)

# Celery is temporarily disabled; using threading fallback
# from celery import shared_task
# @shared_task


def process_document_in_background(doc_id: int) -> None:
    """Background worker for OCR processing."""
    try:
        doc = Document.objects.get(id=doc_id)
        if doc.processed and doc.error_message is None:
            logger.info(f"Document {doc_id} already processed, skipping.")
            return

        # Check if file exists
        if not doc.file or not doc.file.path or not os.path.exists(doc.file.path):
            doc.processed = False
            doc.error_message = "File not found on disk."
            doc.save(update_fields=["processed", "error_message"])
            return

        ocr_result = extract_document_ai(doc.file.path, doc_type=doc.doc_type)
        if not ocr_result or "error" in ocr_result:
            doc.processed = False
            doc.error_message = ocr_result.get("error", "AI extraction failed – no text detected")
            doc.extracted_data = {}
            doc.save(update_fields=["processed", "error_message", "extracted_data"])
            return

        ocr_text, final_data = build_document_text_and_fields(ocr_result)
        doc.extracted_text = ocr_text

        # Auto-detect doc type only if the user left it as the default and
        # nothing schema-based came back (a known doc_type already drove
        # structured extraction above, so don't second-guess it here).
        if (not doc.doc_type or doc.doc_type == "other_document") and len(ocr_text) >= 20:
            doc.doc_type = detect_document_type(ocr_text) or doc.doc_type

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
                processed_at=timezone.now(),
            )
        except Exception as update_error:
            logger.error(f"Failed to update document {doc_id} after error: {update_error}")
