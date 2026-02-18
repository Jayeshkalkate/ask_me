# C:\chatbot\ask_me\core\views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.utils import timezone
from django.db.models import Q
from django.core.paginator import Paginator
import json, tempfile, os, logging, uuid
from rapidfuzz import fuzz
import numpy as np
import base64
from .ocr_utils import extract_text_from_document
from .models import Document, DOCUMENT_FIELD_TEMPLATES
from .forms import DocumentUploadForm, DocumentEditForm
from .ai_utils import detect_document_type, extract_structured_data
from .ocr_utils import (
    process_document_file_enhanced,
    process_document_file,  # legacy support
    batch_process_documents,
    validate_ocr_environment,
    get_supported_document_types,
    is_image_blurry,
    DocumentAnalyzer,
)

from .models import convert_numpy
logger = logging.getLogger(__name__)


# def process_document(request):
#     if request.method == "POST":
#         uploaded_file = request.FILES["document"]

#         # Save temporarily
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             for chunk in uploaded_file.chunks():
#                 temp_file.write(chunk)
#             temp_path = temp_file.name

#         try:
#             # OCR
#             # extracted_text = extract_text_from_file(temp_path)
#             from .ocr_utils import extract_text_from_document
#             extracted_text = extract_text_from_document(temp_path)

#             # AI Extraction
#             structured_data = extract_structured_data(extracted_text)
#             structured_data = convert_numpy(structured_data)

#             # Save only structured data
#             Document.objects.create(
#                 user=request.user,
#                 document_type=structured_data.get("document_type"),
#                 name=structured_data.get("name"),
#                 date_of_birth=structured_data.get("date_of_birth"),
#                 address=structured_data.get("address"),
#                 id_number=structured_data.get("id_number"),
#             )

#         finally:
#             # DELETE FILE IMMEDIATELY
#             if os.path.exists(temp_path):
#                 os.remove(temp_path)

#         return redirect("result_page")

# -------------------------------------------------
# 🔹 HOMEPAGE WITH DASHBOARD
# -------------------------------------------------
@login_required
def homepage(request):
    recent_docs = Document.objects.filter(user=request.user).order_by("-created_at")[:5]

    context = {
        "recent_documents": recent_docs,
        "total_documents": recent_docs.count(),
        "processed_documents": recent_docs.filter(processed=True).count(),
        "failed_documents": recent_docs.filter(error_message__isnull=False).count(),
        "ocr_status": validate_ocr_environment(),
        "supported_doc_types": get_supported_document_types(),
    }

    return render(request, "index.html", context)


# -------------------------------------------------
# 🔹 DOCUMENT LIBRARY
# -------------------------------------------------
@login_required
def document_library(request):
    """Display all user documents with filtering and pagination."""
    documents = Document.objects.filter(user=request.user).order_by("-created_at")
    # documents = Document.objects.filter(user=request.user).order_by("-created_at")[:5]

    # Filtering
    doc_type = request.GET.get("doc_type")
    status = request.GET.get("status")
    search_query = request.GET.get("search")

    if doc_type and doc_type != "all":
        documents = documents.filter(doc_type=doc_type)

    if status == "processed":
        documents = documents.filter(processed=True, error_message__isnull=True)
    elif status == "failed":
        documents = documents.filter(error_message__isnull=False)
    elif status == "pending":
        documents = documents.filter(processed=False, error_message__isnull=True)

    if search_query:
        documents = documents.filter(
            Q(extracted_text__icontains=search_query)
            | Q(file__icontains=search_query)
            | Q(doc_type__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(documents, 10)  # 10 documents per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    context = {
        "page_obj": page_obj,
        "doc_types": get_supported_document_types(),
        "total_docs": documents.count(),
        "current_filters": {
            "doc_type": doc_type,
            "status": status,
            "search": search_query,
        },
    }

    return render(request, "document_library.html", context)


# -------------------------------------------------
# 🔹 ENHANCED EDIT DOCUMENT - WITH DATA CLEANUP
# -------------------------------------------------
@login_required
def edit_document(request, pk):
    """Document editing that replaces extracted_data with user_edited_data"""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        form = DocumentEditForm(request.POST, instance=document)
        if form.is_valid():
            # Get the cleaned user_edited_data
            user_edited_data = form.cleaned_data["user_edited_data"]

            # Use the model method to update and clean data
            document.update_user_data(user_edited_data)

            messages.success(
                request,
                "✅ Document data updated successfully! Original extracted data has been replaced.",
            )
            return redirect("core:document_detail", pk=document.pk)
        else:
            messages.error(request, "❌ Please correct the errors below.")
    else:
        # Initialize with current display data (prioritizes user_edited_data)
        display_data = document.display_data
        form = DocumentEditForm(
            instance=document,
            initial={
                "user_edited_data": json.dumps(
                    display_data, indent=2, ensure_ascii=False
                )
            },
        )

    # Prepare page data for template display
    page_data = []
    display_data = document.display_data

    if display_data:
        for page_key, fields in display_data.items():
            if isinstance(fields, dict):
                page_data.append(
                    {
                        "page_number": page_key,
                        "fields": [
                            (k, v) for k, v in fields.items() if k != "_metadata"
                        ],
                        "metadata": fields.get("_metadata", {}),
                    }
                )

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


# -------------------------------------------------
# 🔹 ENHANCED UPLOAD DOCUMENT (JSON-safe)
# -------------------------------------------------
logger = logging.getLogger(__name__)


# -------------------------------------------------
# 🔹 ENHANCED UPLOAD DOCUMENT (JSON-safe)
# -------------------------------------------------


@login_required
def upload_document(request):

    if request.method == "POST":
        uploaded_file = request.FILES.get("file")

        # -----------------------------------------
        # STEP 0 — VALIDATE FILE
        # -----------------------------------------
        if not uploaded_file:
            messages.error(request, "No file uploaded.")
            return redirect("core:upload")

        document = None

        try:
            # -----------------------------------------
            # STEP 1 — SAVE FILE
            # -----------------------------------------
            document = Document.objects.create(
                user=request.user,
                file=uploaded_file,
                processed=False,
            )

            # Ensure file is saved to disk
            document.refresh_from_db()

            file_path = document.file.path
            print("FILE PATH:", file_path)

            # -----------------------------------------
            # STEP 1.1 — FILE PATH SAFETY CHECK (CRITICAL)
            # -----------------------------------------
            if not file_path or not os.path.exists(file_path):
                print("❌ File path invalid:", file_path)

                document.processed = False
                document.error_message = "File path invalid or file not saved"
                document.extracted_data = {}
                document.save()

                messages.error(request, "File upload failed — file not found.")
                return redirect("core:document_library")

            logger.info(f"Processing file: {file_path}")

            # -----------------------------------------
            # STEP 2 — RUN OCR PIPELINE
            # -----------------------------------------
            extracted_data = process_document_file_enhanced(file_path)

            # -----------------------------------------
            # STEP 3 — STOP IF OCR FAILED
            # -----------------------------------------
            if not extracted_data:
                logger.warning("OCR failed — no data extracted")

                document.processed = False
                document.error_message = "OCR failed — no text detected"
                document.extracted_data = {}
                document.save()

                return render(request, "no_fields.html")
            
            # -----------------------------------------
            # STEP 4 — EXTRACT RAW OCR TEXT (CORRECTED)
            # -----------------------------------------
            
            ocr_text = ""
            
            if isinstance(extracted_data, dict):
                
                for page_key, page_data in extracted_data.items():
                    
                    if not isinstance(page_data, dict):
                        continue
                    
                    # Case 1 — raw_text exists
                    
                    if "raw_text" in page_data:
                        ocr_text += page_data.get("raw_text", "") + " "
                        
                        # Case 2 — structured fields only
                        
                    else:
                        for field, value in page_data.items():
                            if field != "_metadata" and value:
                                ocr_text += str(value) + " "
                                ocr_text = ocr_text.strip()
                                
                                print("\n================ OCR DEBUG ================")
                                print("OCR TEXT LENGTH:", len(ocr_text))
                                print("OCR TEXT SAMPLE:", ocr_text[:500])
                                print("===========================================\n")


            # -----------------------------------------
            # STEP 5 — DETECT DOCUMENT TYPE
            # -----------------------------------------
            if not ocr_text or len(ocr_text) < 20:
                detected_type = "Other_Document"
                print("⚠️ OCR text too small — detection skipped")
            else:
                print("✅ Calling OpenAI detection...")
                detected_type = detect_document_type(ocr_text)

            document.doc_type = detected_type
            logger.info(f"Detected document type: {detected_type}")

            # -----------------------------------------
            # STEP 6 — SAVE CLEAN JSON DATA
            # -----------------------------------------
            clean_data = convert_numpy(extracted_data)

            document.extracted_data = clean_data
            document.processed = True
            document.error_message = None
            document.save()

            messages.success(request, "✅ Document processed successfully!")
            return redirect("core:edit_document", document.id)

        # -----------------------------------------
        # STEP 7 — GLOBAL ERROR HANDLER
        # -----------------------------------------
        except Exception as e:
            logger.exception("Upload processing failed")

            if document:
                document.processed = False
                document.error_message = str(e)
                document.extracted_data = {}
                document.save()

            messages.error(request, f"Processing failed: {str(e)}")
            return redirect("core:upload")

    # -----------------------------------------
    # STEP 8 — GET REQUEST
    # -----------------------------------------
    return render(request, "upload_document.html")


# -------------------------------------------------
# 🔹 BATCH UPLOAD DOCUMENTS
# -------------------------------------------------
@login_required
def batch_upload_documents(request):
    """Handle multiple document uploads at once."""
    if request.method == "POST" and request.FILES.getlist("files"):
        files = request.FILES.getlist("files")
        doc_type = request.POST.get("doc_type", "other_document")

        if len(files) > 10:  # Limit batch size
            messages.error(request, "❌ Maximum 10 files allowed per batch upload.")
            return redirect("core:batch_upload")

        # Create documents first
        documents = []
        for file in files:
            document = Document.objects.create(
                user=request.user,
                file=file,
                doc_type=doc_type,
                uploaded_at=timezone.now(),
            )
            documents.append(document)

        try:
            # Process all documents
            file_paths = [doc.file.path for doc in documents]
            batch_results = batch_process_documents(
                file_paths, [doc_type] * len(documents)
            )

            success_count = 0
            for doc, (filename, result) in zip(documents, batch_results.items()):
                if "error" not in result:
                    # Update document with result (simplified - you'd want full processing here)
                    doc.processed = True
                    clean_data = convert_numpy(result)
                    doc.extracted_data = clean_data
                    doc.save()

                    success_count += 1
                else:
                    doc.error_message = result.get("error", "Unknown error")
                    doc.save()

            if success_count > 0:
                messages.success(
                    request,
                    f"✅ Successfully processed {success_count} out of {len(documents)} documents.",
                )
            if success_count < len(documents):
                messages.warning(
                    request,
                    f"⚠️ {len(documents) - success_count} documents failed processing.",
                )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            messages.error(request, f"❌ Batch processing failed: {str(e)}")

        return redirect("core:document_library")

    return render(
        request,
        "batch_index.html",
        {"supported_doc_types": get_supported_document_types()},
    )


# -------------------------------------------------
# 🔹 ENHANCED DOCUMENT DETAIL VIEW
# -------------------------------------------------
@login_required
def document_detail(request, pk):

    document = get_object_or_404(Document, pk=pk, user=request.user)

    pages_data = []

    for page_key, page_data in document.display_data.items():
        metadata = page_data.get("_metadata", {})
        fields = [(k, v) for k, v in page_data.items() if k != "_metadata"]

        pages_data.append(
            {
                "page_key": page_key,
                "fields": fields,
                "metadata": metadata,
            }
        )

    return render(
        request,
        "document_detail.html",
        {
            "document": document,
            "pages_data": pages_data,
        },
    )


# -------------------------------------------------
# 🔹 REPROCESS DOCUMENT
# -------------------------------------------------
@login_required
def reprocess_document(request, pk):
    """Reprocess a document with different settings."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        new_doc_type = request.POST.get("doc_type", document.doc_type)
        enhancement_level = request.POST.get("enhancement_level", "auto")

        try:
            # For reprocessing, we'd use the enhanced function with specific parameters
            # This is a simplified version - you'd integrate the full reprocessing logic
            ocr_result = process_document_file_enhanced(
                document.file.path, doc_type=new_doc_type, auto_detect=True
            )

            # Update document (similar to upload logic)
            clean_data = convert_numpy(ocr_result)
            document.extracted_data = clean_data
            document.doc_type = new_doc_type
            document.error_message = None
            document.processed = True
            document.last_modified = timezone.now()
            document.save()

            messages.success(request, "✅ Document reprocessed successfully!")

        except Exception as e:
            logger.error(f"Reprocessing failed for document {document.id}: {e}")
            messages.error(request, f"❌ Reprocessing failed: {str(e)}")

        return redirect("core:document_detail", pk=document.id)

    return render(
        request,
        "reprocess_document.html",
        {"document": document, "supported_doc_types": get_supported_document_types()},
    )


# -------------------------------------------------
# 🔹 ENHANCED CHAT API - PRIORITIZE USER EDITED DATA
# -------------------------------------------------
@login_required
@csrf_exempt
def chat_api(request):
    """Enhanced chat API that always uses user_edited_data when available"""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    user_message = data.get("message", "").strip().lower()
    conversation_id = data.get("conversation_id", str(uuid.uuid4()))

    if user_message:
        return handle_chat_query_enhanced(request, user_message, conversation_id)
    else:
        return JsonResponse({"error": "No message provided"}, status=400)


def handle_chat_query_enhanced(request, user_message, conversation_id):
    """Handle chat queries - ALWAYS prioritize user_edited_data"""
    docs = Document.objects.filter(user=request.user, processed=True)

    if not docs.exists():
        return JsonResponse(
            {
                "response": "I couldn't find any processed documents in your library. Please upload some documents first.",
                "conversation_id": conversation_id,
            }
        )

    best_matches = []
    threshold = 65

    for doc in docs:
        # ALWAYS use user_edited_data if available, otherwise extracted_data
        search_data = doc.display_data
        search_text = doc.extracted_text or ""

        if not search_data and not search_text:
            continue

        # Strategy 1: Direct text search
        if search_text:
            text_ratio = fuzz.partial_ratio(user_message, search_text.lower())
            if text_ratio >= threshold:
                is_user_edited = bool(doc.user_edited_data)
                best_matches.append((doc, text_ratio, "direct_match", is_user_edited))

        # Strategy 2: Field-value search in structured data
        if search_data:
            for page_key, page_data in search_data.items():
                if isinstance(page_data, dict):
                    for field_key, field_value in page_data.items():
                        if field_key == "_metadata" or not field_value:
                            continue

                        # Check field name match
                        field_ratio = fuzz.partial_ratio(
                            user_message, field_key.lower()
                        )
                        # Check field value match
                        value_ratio = fuzz.partial_ratio(
                            user_message, str(field_value).lower()
                        )

                        combined_ratio = max(field_ratio, value_ratio)
                        if combined_ratio >= threshold:
                            is_user_edited = bool(doc.user_edited_data)
                            best_matches.append(
                                (
                                    doc,
                                    combined_ratio,
                                    f"structured_match",
                                    field_key,
                                    field_value,
                                    is_user_edited,
                                )
                            )

    # Sort by match score
    best_matches.sort(key=lambda x: x[1], reverse=True)

    if best_matches:
        best_match = best_matches[0]
        doc, score = best_match[0], best_match[1]

        if "direct_match" in best_match[2]:
            # Extract relevant snippet
            is_user_edited = best_match[3]
            response_text = extract_relevant_snippet(user_message, search_text)
            source_indicator = (
                "✏️ (from your edited data)"
                if is_user_edited
                else "📄 (from original extraction)"
            )
            response_text = f"{response_text} {source_indicator}"
        else:
            # Structured data match
            field_key, field_value, is_user_edited = (
                best_match[3],
                best_match[4],
                best_match[5],
            )
            source_indicator = (
                "✏️ From your edited data"
                if is_user_edited
                else "📄 From original extraction"
            )
            response_text = f"{source_indicator}:\n**{field_key}**: {field_value}"

        # Add confidence indicator
        confidence = "high" if score >= 85 else "medium" if score >= 70 else "low"

        return JsonResponse(
            {
                "response": response_text,
                "confidence": confidence,
                "document_id": doc.id,
                "document_type": doc.get_doc_type_display(),
                "data_source": "user_edited" if is_user_edited else "extracted",
                "conversation_id": conversation_id,
            }
        )

    # No good matches found
    suggestion = generate_search_suggestion(user_message, docs)
    return JsonResponse(
        {
            "response": f"I couldn't find specific information matching '{user_message}' in your documents. {suggestion}",
            "conversation_id": conversation_id,
        }
    )


def extract_relevant_snippet(query, text, max_length=200):
    """Extract relevant text snippet around the query match."""
    query_words = query.split()
    text_lower = text.lower()

    # Find the best matching segment
    for word in query_words:
        if len(word) < 4:  # Skip very short words
            continue
        idx = text_lower.find(word)
        if idx != -1:
            start = max(0, idx - 50)
            end = min(len(text), idx + len(word) + 150)
            snippet = text[start:end]
            return f'📄 Found relevant information:\n"...{snippet}..."'

    # Fallback: return beginning of text
    return f"📄 Found in your documents:\n{text[:max_length]}..."


def generate_search_suggestion(query, documents):
    """Generate helpful search suggestions."""
    all_text = " ".join([doc.extracted_text for doc in documents if doc.extracted_text])
    words = all_text.split()

    # Common field names in documents
    common_fields = [
        "name",
        "father_name",
        "mother_name",
        "date",
        "dob",
        "number",
        "id",
        "address",
        "issue",
        "expiry",
        "gender",
        "document_type",
        "place_of_issue",
        "nationality",
        "photo",
        "signature",
        "vehicle_number",
        "vehicle_type",
        "registration_date",
        "engine_number",
        "chassis_number",
    ]

    for field in common_fields:
        if field in query:
            return "Try searching for specific values like 'John Doe' or '123 Main Street' instead of field names."

    return "Try using specific keywords from your documents or ask about particular fields like name, date, or ID number."


def handle_base64_upload(file_base64, file_type):
    """Handle base64 file upload and processing."""
    tmp_file_path = None
    try:
        # Validate base64 data
        if not file_base64.startswith("data:"):
            return JsonResponse({"error": "Invalid base64 format"}, status=400)

        # Extract base64 data
        file_data = file_base64.split("base64,")[1]
        decoded_file = base64.b64decode(file_data)

        # Create temporary file
        file_extension = ".pdf" if file_type == "application/pdf" else ".jpg"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        tmp_file.write(decoded_file)
        tmp_file.flush()
        tmp_file_path = tmp_file.name
        tmp_file.close()

        # Process document
        ocr_result = process_document_file_enhanced(
            tmp_file_path, doc_type=None, auto_detect=True
        )

        # Format response
        response_data = format_ocr_response(ocr_result)

        return JsonResponse(response_data)

    except Exception as e:
        logger.error(f"Base64 upload processing failed: {e}")
        return JsonResponse({"error": f"Processing failed: {str(e)}"}, status=500)
    finally:
        # Cleanup
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def format_ocr_response(ocr_result):
    """Format OCR result for API response."""
    formatted_data = {}
    text_lines = ["📋 Extracted Data:"]
    has_errors = False
    has_warnings = False

    for page_key, page_data in ocr_result.items():
        if "error" in page_data:
            formatted_data[page_key] = {"error": page_data["error"]}
            text_lines.append(f"\n❌ {page_key}: {page_data['error']}")
            has_errors = True
        else:
            # Extract metadata
            # metadata = page_data.pop("_metadata", {})
            metadata = page_data.get("_metadata", {})
            fields = {k: v for k, v in page_data.items() if k != "_metadata"}

            formatted_data[page_key] = {"fields": fields, "metadata": metadata}

            text_lines.append(f"\n📄 {page_key}:")
            for field, value in fields.items():
                if value:
                    text_lines.append(f"  • {field}: {value}")

            # Collect warnings
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


# -------------------------------------------------
# 🔹 DOCUMENT ANALYSIS API
# -------------------------------------------------
@login_required
@csrf_exempt
def analyze_document_quality(request, pk):
    """API endpoint to analyze document quality."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    document = get_object_or_404(Document, pk=pk, user=request.user)

    try:
        analyzer = DocumentAnalyzer()
        quality_scores = analyzer.calculate_image_quality_score(document.file.path)
        is_blurry, blur_score = is_image_blurry(document.file.path)

        return JsonResponse(
            {
                "quality_scores": quality_scores,
                "is_blurry": is_blurry,
                "blur_score": blur_score,
                "recommendations": generate_quality_recommendations(
                    quality_scores, is_blurry
                ),
            }
        )

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def generate_quality_recommendations(quality_scores, is_blurry):
    """Generate quality improvement recommendations."""
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

    return recommendations


# -------------------------------------------------
# 🔹 DELETE DOCUMENT
# -------------------------------------------------
@login_required
def delete_document(request, pk):

    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        document.delete()
        messages.success(request, "✅ Document deleted successfully.")
        return redirect("core:document_library")

    return render(request, "confirm_delete.html", {"document": document})


# -------------------------------------------------
# 🔹 SYSTEM STATUS
# -------------------------------------------------
@login_required
def system_status(request):
    """Display system status and OCR environment information."""
    ocr_status = validate_ocr_environment()

    # Get system statistics
    user_doc_count = Document.objects.filter(user=request.user).count()
    total_doc_count = Document.objects.count()  # Admin only in real implementation

    context = {
        "ocr_status": ocr_status,
        "supported_doc_types": get_supported_document_types(),
        "user_statistics": {
            "documents_uploaded": user_doc_count,
            "documents_processed": Document.objects.filter(
                user=request.user, processed=True
            ).count(),
            "success_rate": calculate_success_rate(request.user),
        },
        "system_ready": all(ocr_status.values()),
    }

    return render(request, "system_status.html", context)


def calculate_success_rate(user):
    """Calculate document processing success rate for user."""
    total = Document.objects.filter(user=user).count()
    if total == 0:
        return 100.0

    successful = Document.objects.filter(
        user=user, processed=True, error_message__isnull=True
    ).count()
    return round((successful / total) * 100, 1)


# -------------------------------------------------
# 🔹 SEARCH DOCUMENT FIELD (MATCHES TEMPLATE)
# -------------------------------------------------
@login_required
def search_document_field(request):
    """
    Search for a specific field or value in documents.
    PRIORITIZES user_edited_data over extracted_data.
    """
    field_query = request.GET.get("q", "").strip().lower()
    results = []

    if not field_query:
        return render(
            request, "index.html", {"error": "Please enter a search term.", "field": ""}
        )

    # Fetch processed documents for the logged-in user
    documents = Document.objects.filter(user=request.user, processed=True)

    for doc in documents:
        # Prioritize user_edited_data over extracted_data for search
        search_data = (
            doc.user_edited_data if doc.user_edited_data else doc.extracted_data
        )

        if not search_data:
            continue

        for page_key, page_data in search_data.items():
            if not isinstance(page_data, dict):
                continue

            for field_key, field_value in page_data.items():
                # Skip metadata or empty fields
                if field_key == "_metadata" or not field_value:
                    continue

                # Match either field name or value
                if (
                    field_query in field_key.lower()
                    or field_query in str(field_value).lower()
                ):
                    # Determine if this is from user_edited_data
                    is_user_edited = doc.user_edited_data is not None
                    data_source = "user_edited" if is_user_edited else "extracted"

                    results.append(
                        {
                            "document_id": doc.id,
                            "doc_type": doc.get_doc_type_display(),
                            "page": page_key,
                            "field": field_key,
                            "value": field_value,
                            "data_source": data_source,
                            "is_user_edited": is_user_edited,
                            # For display purposes - show both values if available
                            "extracted_value": (
                                doc.extracted_data.get(page_key, {}).get(field_key, "")
                                if doc.extracted_data
                                else ""
                            ),
                        }
                    )

    context = {
        "field": field_query,
        "results": results,
        "total_results": len(results),
    }

    # If nothing found
    if not results:
        context["error"] = f'No matches found for "{field_query}".'

    return render(request, "index.html", context)
