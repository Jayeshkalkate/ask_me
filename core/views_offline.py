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
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import UploadedFile

from .models import Document, convert_numpy
from .ocr_utils import process_document_file_enhanced
from .ai_utils import extract_structured_data, detect_document_type, clean_ocr_text

logger = logging.getLogger(__name__)

# @csrf_exempt removed – CSRF protection is now enforced
@login_required
def offline_upload(request):
    """Handle document upload for offline mode"""
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
            
            # Process document with OCR
            ocr_result = process_document_file_enhanced(tmp_path, doc_type=doc_type, auto_detect=True)
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            
            # Extract structured data
            extracted_data = {}
            extracted_text = ""
            
            if isinstance(ocr_result, dict):
                for page_key, page_data in ocr_result.items():
                    # Skip internal keys (starting with "_") like "_summary"
                    if page_key.startswith("_") or not isinstance(page_data, dict):
                        continue
                    
                    # Get raw text
                    if 'raw_text' in page_data:
                        extracted_text += page_data['raw_text'] + " "
                    
                    # Clean data - remove internal keys
                    cleaned_page = {}
                    for key, value in page_data.items():
                        if key not in ['_metadata', 'status', 'raw_text', 'structured_data'] and value:
                            if isinstance(value, str):
                                cleaned_page[key] = value.strip()
                            else:
                                cleaned_page[key] = value
                    
                    if cleaned_page:
                        extracted_data[page_key] = cleaned_page
            
            # Clean extracted text
            extracted_text = clean_ocr_text(extracted_text)
            
            # Auto-detect document type if not provided
            if not doc_type or doc_type == 'other_document':
                detected = detect_document_type(extracted_text)
                if detected and detected != 'Other_Document':
                    doc_type = detected.lower().replace(' ', '_')
                else:
                    doc_type = 'other_document'
            
            # If no data was extracted, try fallback extraction
            if not extracted_data and extracted_text:
                # Try structured extraction from text
                structured = extract_structured_data(extracted_text)
                if structured:
                    extracted_data = {'page_1': structured}
            
            # Create response with document data
            document_data = {
                'file_name': uploaded_file.name,
                'file_size': uploaded_file.size,
                'doc_type': doc_type,
                'doc_type_display': dict(Document.DOC_TYPES).get(doc_type, doc_type.replace('_', ' ').title()),
                'extracted_text': extracted_text[:5000] if extracted_text else '',  # Limit text length
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
            return JsonResponse({'error': f'Processing failed: {str(e)}'}, status=500)
            
    except Exception as e:
        logger.error(f"Offline upload error: {e}")
        return JsonResponse({'error': str(e)}, status=500)

# @csrf_exempt removed – CSRF protection now applies
@login_required
def offline_documents_api(request):
    """API endpoint for offline document management"""
    if request.method == 'GET':
        # Return empty list - documents are stored locally
        return JsonResponse({
            'documents': [],
            'message': 'Documents are stored locally in offline mode'
        })
    
    if request.method == 'DELETE':
        # Get document ID from request
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
        # Update document
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
