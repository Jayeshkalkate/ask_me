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
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import UploadedFile

from .models import Document, convert_numpy, validate_file_extension
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

        # Validate file size (10MB max) and extension - matches the
        # classic upload_document view's validation, which this endpoint
        # never applied before (see validate_file_size/validate_file_extension
        # in core/models.py, used by Document's own field validators).
        if uploaded_file.size > 10 * 1024 * 1024:
            return JsonResponse({'error': 'File size must be under 10MB'}, status=400)
        try:
            validate_file_extension(uploaded_file)
        except ValidationError as e:
            return JsonResponse({'error': ' '.join(e.messages)}, status=400)

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
            extracted_text, extracted_data = build_document_text_and_fields(ocr_result, doc_type=doc_type)
            extracted_text = clean_ocr_text(extracted_text)

            # Known doc type but no structured fields came back (Gemini's
            # structured call failed, e.g. a transient 503) - tell the
            # client plainly instead of quietly syncing an empty/guessed
            # result, so it can show a "needs reprocessing" state.
            from .models import DOCUMENT_FIELD_TEMPLATES
            page_1_fields = (extracted_data or {}).get("page_1") or {}
            extraction_incomplete = bool(
                doc_type in DOCUMENT_FIELD_TEMPLATES and not page_1_fields
            )

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
                'extraction_incomplete': extraction_incomplete,
                'processed_at': timezone.now().isoformat(),
                'created_at': timezone.now().isoformat()
            }

            # Persist a real Document row - this used to be skipped
            # entirely, so anything uploaded through this endpoint (the
            # chat page's quick-upload widget, and documents queued while
            # offline once they sync back) was extracted successfully but
            # then silently thrown away server-side: it never appeared in
            # the Document Library, couldn't be shared/exported, and was
            # permanently lost if the browser's IndexedDB was ever
            # cleared. The uploaded file's read pointer was already
            # consumed above (chunks() while writing the temp file for
            # extraction), so it must be rewound before Django's storage
            # backend can read it again to save it.
            try:
                uploaded_file.seek(0)
            except (AttributeError, ValueError):
                pass
            document = Document.objects.create(
                user=request.user,
                file=uploaded_file,
                doc_type=doc_type,
                extracted_text=extracted_text,
                extracted_data=convert_numpy(extracted_data),
                processed=True,
                processed_at=timezone.now(),
                error_message=(
                    "AI extraction service didn't return structured fields for this "
                    "document (likely a temporary outage) - reprocess it once the "
                    "service has recovered."
                    if extraction_incomplete else None
                ),
            )
            document_data['id'] = document.id
            document_data['created_at'] = document.created_at.isoformat()
            document_data['processed_at'] = document.processed_at.isoformat()

            message = (
                "AI extraction service didn't return structured fields for this "
                "document (likely a temporary outage) - reprocess it once you're "
                "back online and the service has recovered."
                if extraction_incomplete else
                'Document processed successfully offline'
            )
            return JsonResponse({
                'success': True,
                'document': document_data,
                'message': message,
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

    GET: returns every one of the current user's documents - including
    their extracted_data/user_edited_data field values - as JSON, so the
    browser can cache them all in IndexedDB (see static/js/db.js and the
    sync call in static/js/pwa.js). Without this, the offline chatbot
    could only ever answer questions about documents that happened to be
    uploaded through the chat page's own upload widget while online - any
    document added via the regular "Upload Document" page, or fields
    edited on the document-detail page, were invisible to it. This makes
    every one of the user's documents (and their latest edits) available
    for offline lookup, not just ones uploaded through one specific path.
    DELETE/PUT: deletes/updates a document locally (handled client-side);
    unchanged from before.
    """
    if request.method == 'GET':
        documents = Document.objects.filter(user=request.user).order_by('-created_at')
        docs_payload = []
        for doc in documents:
            docs_payload.append({
                'id': doc.id,
                'file_name': doc.original_filename or (
                    os.path.basename(doc.file.name) if doc.file else ''
                ),
                'doc_type': doc.doc_type,
                'doc_type_display': doc.get_doc_type_display(),
                # Both are sent so the client can apply the same "edited
                # data wins over originally extracted data" rule the
                # server itself uses (see DocumentManager.get_display_data).
                'extracted_data': convert_numpy(doc.extracted_data) if doc.extracted_data else {},
                'user_edited_data': convert_numpy(doc.user_edited_data) if doc.user_edited_data else None,
                'extracted_text': (doc.extracted_text or '')[:5000],
                'processed': doc.processed,
                'error_message': doc.error_message or '',
                'created_at': doc.created_at.isoformat() if doc.created_at else None,
                'processed_at': doc.processed_at.isoformat() if doc.processed_at else None,
            })
        return JsonResponse({'documents': docs_payload})

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

            # This used to be a stub that always claimed success without
            # writing anything - so edits made while OFFLINE (queued as a
            # pending 'UPDATE_DOCUMENT' op by templates/edit_document.html,
            # sent here once back online by static/js/pwa.js::updateDocument)
            # were silently discarded from the server, and the very next
            # pull-sync would overwrite the local copy with the server's
            # unedited data too - a double loss of the user's edit.
            document = get_object_or_404(Document, pk=doc_id, user=request.user)

            if 'user_edited_data' in update_data:
                if not isinstance(update_data['user_edited_data'], dict):
                    return JsonResponse(
                        {'error': 'user_edited_data must be a JSON object'}, status=400
                    )
                document.update_user_data(update_data['user_edited_data'])

            return JsonResponse({
                'success': True,
                'message': f'Document {doc_id} updated',
                'data': update_data
            })
        except Document.DoesNotExist:
            return JsonResponse({'error': 'Document not found'}, status=404)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
