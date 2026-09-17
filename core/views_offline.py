# core/views_offline.py - Offline-first document handling
import json
import logging
import tempfile
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from django.core.files.uploadedfile import UploadedFile

from .models import Document, convert_numpy
from .ai_extract import extract_document_ai
from .ai_utils import detect_document_type, clean_ocr_text
from .utils import build_document_text_and_fields

logger = logging.getLogger(__name__)


@login_required
def offline_upload(request):
    """
    Endpoint the client syncs queued documents to once back online (see
    static/js/offline-processor.js::syncPendingDocuments). Files uploaded
    while offline are NOT processed on-device anymore - they're just saved
    locally and pushed here as soon as connectivity returns, so AI
    extraction (Gemini, same as the normal online upload) can run and
    return real structured data.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    try:
        uploaded_file = request.FILES.get('file')
        doc_type = request.POST.get('doc_type', 'other_document')

        if not uploaded_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        # Validate file size (10MB max)
        if uploaded_file.size > 10 * 1024 * 1024:
            return JsonResponse({'error': 'File size must be under 10MB'}, status=400)

        # Process file immediately
        try:
            # Create temp file
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Extract text using the free Gemini AI tier (Tesseract disabled)
            ocr_result = extract_document_ai(
                tmp_path,
                doc_type=doc_type,
                auto_detect=True
            )

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

            if isinstance(ocr_result, dict) and "error" in ocr_result and len(ocr_result) == 1:
                return JsonResponse({'error': ocr_result["error"]}, status=502)

            # Prefer the AI's own structured fields (extracted directly
            # against doc_type's predefined schema in ai_extract.py) over
            # the legacy regex-based fallback - see core/utils.py.
            extracted_text, extracted_data = build_document_text_and_fields(ocr_result)
            extracted_text = clean_ocr_text(extracted_text)

            # Auto-detect document type only if the user didn't pick one
            # (a known doc_type already drove structured extraction above).
            if not doc_type or doc_type == 'other_document':
                detected = detect_document_type(extracted_text)
                if detected and detected != 'Other_Document':
                    doc_type = detected.lower().replace(' ', '_')
                else:
                    doc_type = 'other_document'

            # Create response with document data
            document_data = {
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'doc_type': doc_type,
                'doc_type_display': dict(Document.DOC_TYPES).get(
                    doc_type, doc_type.replace('_', ' ').title()
                ),
                'extracted_text': extracted_text[:5000] if extracted_text else '',
                'extracted_data': extracted_data,
                'processed': True,
                'processed_at': timezone.now().isoformat(),
                'created_at': timezone.now().isoformat()
            }

            return JsonResponse({
                'success': True,
                'document': document_data,
                'message': 'Document processed successfully offline'
            })

        except Exception as e:
            logger.error(f"Offline processing failed: {e}")
            return JsonResponse(
                {'error': f'Processing failed: {str(e)}'},
                status=500
            )

    except Exception as e:
        logger.error(f"Offline upload error: {e}")
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def offline_documents_api(request):
    """
    API endpoint for offline document management.
    GET: returns empty list (documents stored locally)
    DELETE: deletes a document locally (handled client-side)
    PUT: updates a document locally (handled client-side)
    """
    if request.method == 'GET':
        # Return empty list - documents are stored locally
        return JsonResponse({
            'documents': [],
            'message': 'Documents are stored locally in offline mode'
        })

    if request.method == 'DELETE':
        try:
            data = json.loads(request.body)
            doc_id = data.get('id')
            if not doc_id:
                return JsonResponse({'error': 'Document ID required'}, status=400)

            # Document deletion handled locally
            return JsonResponse({
                'success': True,
                'message': f'Document {doc_id} deleted locally'
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    if request.method == 'PUT':
        try:
            data = json.loads(request.body)
            doc_id = data.get('id')
            update_data = data.get('data', {})

            if not doc_id:
                return JsonResponse({'error': 'Document ID required'}, status=400)

            return JsonResponse({
                'success': True,
                'message': f'Document {doc_id} updated locally',
                'data': update_data
            })
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
