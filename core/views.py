import json
import logging
import tempfile
import threading
import uuid
import base64
import os
from typing import Dict, List, Any, Optional

# Celery is temporarily disabled - using threading fallback
# from .tasks import process_document_in_background

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseRedirect, Http404
from django.views.decorators.csrf import csrf_exempt, csrf_protect, ensure_csrf_cookie
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q, Count
from django.core.paginator import Paginator
from django.urls import reverse

from rapidfuzz import fuzz

# Import from utils
from .utils import clean_extracted_data, get_structured_fields_from_text, INTERNAL_KEYS

from .models import Document, convert_numpy
from .forms import DocumentEditForm
from .ai_utils import detect_document_type, extract_structured_data, generic_extraction
from .ocr_utils import (
    process_document_file_enhanced,
    batch_process_documents,
    validate_ocr_environment,
    get_supported_document_types,
    is_image_blurry,
    DocumentAnalyzer,
)

logger = logging.getLogger(__name__)


# ============================================================
#  HOMEPAGE
# ============================================================
@login_required
def homepage(request):
    """Dashboard showing recent documents and statistics."""
    recent_docs = Document.objects.filter(user=request.user).order_by("-created_at")[:5]
    total_processed = Document.objects.filter(user=request.user, processed=True).count()
    context = {
        "recent_docs": recent_docs,
        "processed_documents": total_processed,
    }
    return render(request, "index.html", context)


# ============================================================
#  DOCUMENT LIBRARY (with filtering & pagination)
# ============================================================
@login_required
def document_library(request):
    """Display user documents with filters and pagination."""
    qs = Document.objects.filter(user=request.user).order_by("-created_at")

    doc_type = request.GET.get("doc_type")
    status = request.GET.get("status")
    search_query = request.GET.get("search")

    if doc_type and doc_type != "all":
        qs = qs.filter(doc_type=doc_type)

    if status == "processed":
        qs = qs.filter(processed=True, error_message__isnull=True)
    elif status == "failed":
        qs = qs.filter(error_message__isnull=False)
    elif status == "pending":
        qs = qs.filter(processed=False, error_message__isnull=True)

    if search_query:
        qs = qs.filter(
            Q(extracted_text__icontains=search_query)
            | Q(file__icontains=search_query)
            | Q(doc_type__icontains=search_query)
        )

    paginator = Paginator(qs, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # Get supported doc types from ocr_utils
    try:
        doc_types = get_supported_document_types()
    except:
        doc_types = []

    context = {
        "page_obj": page_obj,
        "doc_types": doc_types,
        "total_docs": qs.count(),
        "current_filters": {
            "doc_type": doc_type,
            "status": status,
            "search": search_query,
        },
    }
    return render(request, "document_library.html", context)


def _process_document_thread(doc_id: int) -> None:
    """Thread-based background worker for OCR processing (fallback when Celery is disabled)."""
    try:
        # Import the processing function from tasks
        from .tasks import process_document_in_background
        process_document_in_background(doc_id)
    except Exception as e:
        logger.exception(f"Thread processing failed for doc {doc_id}")


# ============================================================
#  UPLOAD DOCUMENT
# ============================================================
@login_required
def upload_document(request):
    """Handle single document upload with background processing."""
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            messages.error(request, "No file uploaded.")
            return redirect("core:upload")

        document = Document.objects.create(
            user=request.user,
            file=uploaded_file,
            processed=False,
        )

        if document.extracted_data or document.extracted_text:
            messages.info(request, "Document already processed.")
            return redirect("core:edit_document", pk=document.id)

        # Use threading instead of Celery (Celery is temporarily disabled)
        thread = threading.Thread(target=_process_document_thread, args=(document.id,))
        thread.daemon = True
        thread.start()

        messages.success(
            request,
            "✅ File uploaded. Processing will complete shortly. "
            "You can check the status in the library."
        )
        return redirect("core:document_library")

    return render(request, "upload_document.html")


# ============================================================
#  BATCH UPLOAD DOCUMENT
# ============================================================
@login_required
def batch_upload_documents(request):
    """Upload multiple documents at once (max 10)."""
    if request.method == "POST":
        files = request.FILES.getlist("files")
        if not files:
            messages.error(request, "No files selected.")
            return redirect("core:batch_upload")

        if len(files) > 10:
            messages.error(request, "❌ Maximum 10 files allowed per batch upload.")
            return redirect("core:batch_upload")

        doc_type = request.POST.get("doc_type", "other_document")
        created_docs = []
        for file in files:
            doc = Document.objects.create(
                user=request.user,
                file=file,
                doc_type=doc_type,
                processed=False,
            )
            created_docs.append(doc)

        # Use threading for all documents
        for doc in created_docs:
            thread = threading.Thread(target=_process_document_thread, args=(doc.id,))
            thread.daemon = True
            thread.start()

        messages.success(
            request,
            f"✅ {len(created_docs)} files uploaded. They will be processed in the background."
        )
        return redirect("core:document_library")

    try:
        doc_types = get_supported_document_types()
    except:
        doc_types = []
        
    return render(
        request,
        "batch_index.html",
        {"supported_doc_types": doc_types},
    )


# ============================================================
#  DOCUMENT DETAIL
# ============================================================
@login_required
def document_detail(request, pk):
    """Show document details and extracted data."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    pages_data = []
    display_data = document.display_data
    if display_data:
        for page_key, page_data in display_data.items():
            if isinstance(page_data, dict):
                metadata = page_data.get("_metadata", {})
                # Skip internal keys
                fields = [(k, v) for k, v in page_data.items() if k not in INTERNAL_KEYS and not k.startswith("_")]
                pages_data.append({
                    "page_key": page_key,
                    "fields": fields,
                    "metadata": metadata,
                })

    has_user_edits = bool(document.user_edited_data)
    can_reprocess = True

    return render(request, "document_detail.html", {
        "document": document,
        "pages_data": pages_data,
        "has_user_edits": has_user_edits,
        "can_reprocess": can_reprocess,
    })


# ============================================================
#  EDIT DOCUMENT
# ============================================================
@login_required
def edit_document(request, pk):
    """Edit document JSON data."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        form = DocumentEditForm(request.POST, instance=document)
        if form.is_valid():
            user_edited_data = form.cleaned_data["user_edited_data"]
            document.update_user_data(user_edited_data)
            messages.success(
                request,
                "✅ Document data updated successfully! Original extracted data has been replaced."
            )
            return redirect("core:document_detail", pk=document.pk)
        else:
            messages.error(request, "❌ Please correct the errors below.")
    else:
        form = DocumentEditForm(instance=document)

    # Build page data for visual editor – skip internal keys
    page_data = []
    display_data = document.display_data
    if display_data:
        for page_key, fields in display_data.items():
            if isinstance(fields, dict):
                page_data.append({
                    "page_number": page_key,
                    "fields": [(k, v) for k, v in fields.items() if k not in INTERNAL_KEYS and not k.startswith("_")],
                    "metadata": fields.get("_metadata", {}),
                })

    context = {
        "document": document,
        "form": form,
        "page_data": page_data,
        "has_quality_issues": any(
            page.get("metadata", {}).get("warnings", []) for page in page_data
        ),
        "has_user_edits": document.is_edited,
    }
    return render(request, "edit_document.html", context)


# ============================================================
#  REPROCESS DOCUMENT
# ============================================================
@login_required
def reprocess_document(request, pk):
    """Reprocess a document with new settings."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        new_doc_type = request.POST.get("doc_type", document.doc_type)

        try:
            ocr_result = process_document_file_enhanced(
                document.file.path,
                doc_type=new_doc_type,
                auto_detect=True,
            )
            if "error" in ocr_result:
                raise Exception(ocr_result["error"])

            # Build OCR text
            ocr_parts = []
            for page_data in ocr_result.values():
                if isinstance(page_data, dict):
                    if "raw_text" in page_data:
                        ocr_parts.append(page_data["raw_text"])
                    else:
                        for field, value in page_data.items():
                            if field not in INTERNAL_KEYS and value:
                                ocr_parts.append(str(value))
            ocr_text = " ".join(ocr_parts).strip()

            # Extract structured fields
            structured_data = get_structured_fields_from_text(ocr_text)
            if structured_data:
                final_data = {"page_1": structured_data}
            else:
                # Fallback: clean the whole result
                cleaned = clean_extracted_data(ocr_result)
                if cleaned and isinstance(cleaned, dict):
                    first_page = next(iter(cleaned.values())) if cleaned else {}
                    if first_page and isinstance(first_page, dict):
                        final_data = {"page_1": first_page}
                    else:
                        final_data = {"page_1": {"Content": ocr_text[:500]}}
                else:
                    final_data = {"page_1": {"Content": ocr_text[:500]}}

            document.extracted_data = convert_numpy(final_data)
            document.extracted_text = ocr_text
            document.doc_type = new_doc_type or document.doc_type
            document.error_message = None
            document.processed = True
            document.processed_at = timezone.now()
            document.save()

            messages.success(request, "✅ Document reprocessed successfully!")
        except Exception as e:
            logger.error(f"Reprocessing failed for document {document.id}: {e}")
            messages.error(request, f"❌ Reprocessing failed: {str(e)[:200]}")

        return redirect("core:document_detail", pk=document.pk)

    try:
        doc_types = get_supported_document_types()
    except:
        doc_types = []
        
    return render(
        request,
        "reprocess_document.html",
        {
            "document": document,
            "supported_doc_types": doc_types,
        },
    )


# ============================================================
#  DELETE DOCUMENT
# ============================================================
@login_required
def delete_document(request, pk):
    """Delete a document after confirmation."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        document.delete()
        messages.success(request, "✅ Document deleted successfully.")
        return redirect("core:document_library")

    return render(request, "confirm_delete.html", {"document": document})


# ============================================================
#  SEARCH DOCUMENT FIELD (used in chat interface)
# ============================================================
@login_required
def search_document_field(request):
    query = request.GET.get("q", "").strip()
    if not query:
        return render(request, "search_results.html", {"error": "Please enter a search term.", "query": ""})

    documents = Document.objects.filter(user=request.user, processed=True)
    threshold = 60

    aggregated_results = []

    for doc in documents:
        doc_matches = {
            "document": doc,
            "text_matches": [],
            "field_matches": [],
            "best_score": 0
        }

        # Search in raw text
        if doc.extracted_text:
            text_score = fuzz.partial_ratio(query.lower(), doc.extracted_text.lower())
            if text_score >= threshold:
                doc_matches["text_matches"].append({"score": text_score, "snippet": doc.extracted_text[:300]})
                doc_matches["best_score"] = max(doc_matches["best_score"], text_score)

        # Search in structured fields
        search_data = doc.user_edited_data if doc.user_edited_data else doc.extracted_data
        if search_data:
            for page_key, page_data in search_data.items():
                if not isinstance(page_data, dict):
                    continue
                for field_key, field_value in page_data.items():
                    if field_key.startswith("_") or field_key in ("_metadata", "status", "raw_text", "structured_data"):
                        continue
                    if not field_value:
                        continue
                    name_score = fuzz.partial_ratio(query.lower(), field_key.lower())
                    val_score = fuzz.partial_ratio(query.lower(), str(field_value).lower())
                    best = max(name_score, val_score)
                    if best >= threshold:
                        doc_matches["field_matches"].append({
                            "field": field_key,
                            "value": field_value,
                            "score": best,
                            "page": page_key
                        })
                        doc_matches["best_score"] = max(doc_matches["best_score"], best)

        if doc_matches["best_score"] > 0:
            # Sort field matches by score descending
            doc_matches["field_matches"].sort(key=lambda x: x["score"], reverse=True)
            # Keep only top 5 field matches to avoid cluttering
            doc_matches["field_matches"] = doc_matches["field_matches"][:5]
            aggregated_results.append(doc_matches)

    # Sort aggregated results by best_score descending
    aggregated_results.sort(key=lambda x: x["best_score"], reverse=True)

    # Paginate
    paginator = Paginator(aggregated_results, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "query": query,
        "page_obj": page_obj,
        "total_results": len(aggregated_results),
        "error": f'No matches found for "{query}".' if not aggregated_results else None,
    }
    return render(request, "search_results.html", context)


# ============================================================
#  CHAT API (AI‑powered query)
# ============================================================
@login_required
@csrf_protect
def chat_api(request):
    """API endpoint for document Q&A."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_message = data.get("message", "").strip()
    conversation_id = data.get("conversation_id", str(uuid.uuid4()))

    if not user_message:
        return JsonResponse({"error": "No message provided"}, status=400)

    return _handle_chat_query(request, user_message, conversation_id)


def _handle_chat_query(request, user_message, conversation_id):
    """Core chat logic – returns JsonResponse."""
    docs = Document.objects.filter(user=request.user, processed=True)
    if not docs.exists():
        return JsonResponse({
            "response": "I couldn't find any processed documents in your library. Please upload some documents first.",
            "conversation_id": conversation_id,
        })

    best_matches = []
    threshold = 65
    user_message_lower = user_message.lower()

    for doc in docs:
        search_data = doc.display_data
        search_text = doc.extracted_text or ""

        if search_text:
            ratio = fuzz.partial_ratio(user_message_lower, search_text.lower())
            if ratio >= threshold:
                best_matches.append((doc, ratio, "direct_match", bool(doc.user_edited_data)))

        if search_data:
            for page_data in search_data.values():
                if not isinstance(page_data, dict):
                    continue
                for field_key, field_value in page_data.items():
                    if field_key.startswith("_") or field_key in INTERNAL_KEYS or not field_value:
                        continue
                    field_ratio = fuzz.partial_ratio(user_message_lower, field_key.lower())
                    value_ratio = fuzz.partial_ratio(user_message_lower, str(field_value).lower())
                    combined = max(field_ratio, value_ratio)
                    if combined >= threshold:
                        best_matches.append(
                            (doc, combined, "structured_match", field_key, field_value, bool(doc.user_edited_data))
                        )

    if not best_matches:
        suggestion = _generate_search_suggestion(user_message, docs)
        return JsonResponse({
            "response": f"I couldn't find specific information matching '{user_message}' in your documents. {suggestion}",
            "conversation_id": conversation_id,
        })

    best_matches.sort(key=lambda x: x[1], reverse=True)
    best = best_matches[0]

    if best[2] == "direct_match":
        doc, score, _, is_user_edited = best
        snippet = _extract_relevant_snippet(user_message, search_text)
        source = "✏️ (from your edited data)" if is_user_edited else "📄 (from original extraction)"
        response_text = f"{snippet} {source}"
        doc_id = doc.id
        doc_type = doc.get_doc_type_display()
        data_source = "user_edited" if is_user_edited else "extracted"
    else:
        doc, score, _, field_key, field_value, is_user_edited = best
        source = "✏️ From your edited data" if is_user_edited else "📄 From original extraction"
        response_text = f"{source}:\n**{field_key}**: {field_value}"
        doc_id = doc.id
        doc_type = doc.get_doc_type_display()
        data_source = "user_edited" if is_user_edited else "extracted"

    confidence = "high" if score >= 85 else "medium" if score >= 70 else "low"

    return JsonResponse({
        "response": response_text,
        "confidence": confidence,
        "document_id": doc_id,
        "document_type": doc_type,
        "data_source": data_source,
        "conversation_id": conversation_id,
    })


def _extract_relevant_snippet(query: str, text: str, max_length: int = 200) -> str:
    """Extract a snippet around the query."""
    query_lower = query.lower()
    text_lower = text.lower()
    for word in query_lower.split():
        if len(word) < 4:
            continue
        idx = text_lower.find(word)
        if idx != -1:
            start = max(0, idx - 50)
            end = min(len(text), idx + len(word) + 150)
            return f'📄 Found relevant information:\n"...{text[start:end]}..."'
    return f"📄 Found in your documents:\n{text[:max_length]}..."


def _generate_search_suggestion(query: str, documents) -> str:
    """Suggest alternative search terms."""
    common_fields = [
        "name", "father_name", "mother_name", "date", "dob", "number", "id",
        "address", "issue", "expiry", "gender", "document_type", "place_of_issue",
        "nationality", "photo", "signature", "vehicle_number", "vehicle_type",
        "registration_date", "engine_number", "chassis_number",
    ]
    for field in common_fields:
        if field in query:
            return "Try searching for specific values like 'John Doe' or '123 Main Street' instead of field names."
    return "Try using specific keywords from your documents or ask about particular fields like name, date, or ID number."


# ============================================================
#  BASE64 UPLOAD (for API clients)
# ============================================================
@login_required
@csrf_protect
def handle_base64_upload(request):
    """Handle base64 file upload and return OCR results."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        file_base64 = data.get("file")
        file_type = data.get("type", "application/octet-stream")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not file_base64:
        return JsonResponse({"error": "Missing file data"}, status=400)

    tmp_file_path = None
    try:
        if not file_base64.startswith("data:"):
            return JsonResponse({"error": "Invalid base64 format"}, status=400)

        file_data = file_base64.split("base64,")[1]
        decoded = base64.b64decode(file_data)

        ext = ".pdf" if "pdf" in file_type else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(decoded)
            tmp_file_path = tmp.name

        ocr_result = process_document_file_enhanced(tmp_file_path, doc_type=None, auto_detect=True)
        response = _format_ocr_response(ocr_result)
        return JsonResponse(response)

    except Exception as e:
        logger.error(f"Base64 upload processing failed: {e}")
        return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass


def _format_ocr_response(ocr_result: Dict) -> Dict:
    """Format OCR result for API response."""
    text_lines = ["📋 Extracted Data:"]
    has_errors = False
    has_warnings = False
    formatted_data = {}

    for page_key, page_data in ocr_result.items():
        if "error" in page_data:
            formatted_data[page_key] = {"error": page_data["error"]}
            text_lines.append(f"\n❌ {page_key}: {page_data['error']}")
            has_errors = True
        else:
            metadata = page_data.get("_metadata", {})
            fields = {k: v for k, v in page_data.items() if k not in INTERNAL_KEYS and not k.startswith("_")}
            formatted_data[page_key] = {"fields": fields, "metadata": metadata}
            text_lines.append(f"\n📄 {page_key}:")
            for field, value in fields.items():
                if value:
                    text_lines.append(f"  • {field}: {value}")
            if metadata.get("warnings"):
                has_warnings = True
                for warning in metadata["warnings"]:
                    text_lines.append(f"  ⚠️ {warning}")

    message = "✅ Document processed successfully!"
    if has_errors:
        message = "⚠️ Document processed with some errors."
    elif has_warnings:
        message = "⚠️ Document processed with quality warnings."

    return {
        "text": "\n".join(text_lines),
        "structured_data": formatted_data,
        "message": message,
        "has_errors": has_errors,
        "has_warnings": has_warnings,
    }


# ============================================================
#  DOCUMENT QUALITY ANALYSIS (API)
# ============================================================
@login_required
@csrf_protect
def analyze_document_quality(request, pk):
    """Analyze image quality of a document."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    document = get_object_or_404(Document, pk=pk, user=request.user)
    try:
        analyzer = DocumentAnalyzer()
        quality_scores = analyzer.calculate_image_quality_score(document.file.path)
        is_blurry, blur_score = is_image_blurry(document.file.path)

        recommendations = []
        if is_blurry or quality_scores.get("blur_score", 0) < 0.3:
            recommendations.append("Use a brighter environment with better focus")
        if quality_scores.get("contrast_score", 0) < 0.4:
            recommendations.append("Improve lighting to increase contrast")
        if quality_scores.get("brightness_score", 0) < 0.4:
            recommendations.append("Increase brightness or use flash")
        elif quality_scores.get("brightness_score", 0) > 0.8:
            recommendations.append("Reduce brightness to avoid overexposure")
        if quality_scores.get("overall_score", 0) < 0.6:
            recommendations.append("Retake the photo with steady hands and good lighting")

        return JsonResponse({
            "quality_scores": quality_scores,
            "is_blurry": is_blurry,
            "blur_score": blur_score,
            "recommendations": recommendations,
        })
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return JsonResponse({"error": str(e)}, status=500)


# ============================================================
#  SYSTEM STATUS
# ============================================================
@login_required
def system_status(request):
    """Display system OCR readiness and user statistics."""
    ocr_status = validate_ocr_environment()
    user_doc_count = Document.objects.filter(user=request.user).count()
    user_processed = Document.objects.filter(user=request.user, processed=True).count()
    success_rate = 100.0
    if user_doc_count > 0:
        success_rate = round((user_processed / user_doc_count) * 100, 1)

    try:
        doc_types = get_supported_document_types()
    except:
        doc_types = []

    context = {
        "ocr_status": ocr_status,
        "supported_doc_types": doc_types,
        "user_statistics": {
            "documents_uploaded": user_doc_count,
            "documents_processed": user_processed,
            "success_rate": success_rate,
        },
        "system_ready": all(ocr_status.values()),
    }
    return render(request, "system_status.html", context)
