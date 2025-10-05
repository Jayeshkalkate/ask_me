# C:\chatbot\ask_me\core\views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import json, base64, tempfile, os, logging
from rapidfuzz import fuzz

from .forms import DocumentUploadForm
from .models import Document, DOCUMENT_FIELD_TEMPLATES
from .ocr_utils import process_document_file

# Set up logging
logger = logging.getLogger(__name__)


@login_required
def homepage(request):
    """Render the home page."""
    return render(request, "index.html")


@login_required
def edit_document(request, pk):
    """Edit extracted data (single or multi-page)."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        updated_data = {}
        for key, value in request.POST.items():
            if key.startswith("field_key_"):
                counter = key.replace("field_key_", "")
                field_value = request.POST.get(f"field_value_{counter}", "")
                # Flatten everything under page_1
                if "page_1" not in updated_data:
                    updated_data["page_1"] = {}
                updated_data["page_1"][value.strip()] = field_value.strip()

        # Save back
        document.extracted_data = updated_data

        # Rebuild extracted_text for search/chat
        lines = []
        for page, fields in updated_data.items():
            for k, v in fields.items():
                lines.append(f"{k}: {v}")
        document.extracted_text = "\n".join(lines)

        document.save()
        messages.success(request, "✅ Extracted data updated successfully.")
        return redirect("core:document_detail", pk=document.pk)

    return render(request, "edit_document.html", {"document": document})


@login_required
def upload_document(request):
    """Handle document upload and OCR extraction with template integration."""
    if request.method == "POST" and request.FILES.get("file"):
        doc_file = request.FILES["file"]
        doc_type = request.POST.get("doc_type", "other_document")

        # ✅ Get template for this document type
        template = DOCUMENT_FIELD_TEMPLATES.get(
            doc_type, DOCUMENT_FIELD_TEMPLATES["other_document"]
        ).copy()  # Important: copy to avoid modifying original

        # ✅ Create document with template as initial extracted_data
        document = Document.objects.create(
            user=request.user,
            file=doc_file,
            doc_type=doc_type,
            extracted_data={"page_1": template},  # Wrap in page_1 for consistency
        )

        try:
            # ✅ Process OCR and merge with template
            ocr_result = process_document_file(document.file.path, doc_type)

            # ✅ Ensure OCR result is properly structured
            if not ocr_result:
                ocr_result = {"page_1": {}}
            elif isinstance(ocr_result, str):
                ocr_result = {"page_1": {"raw_text": ocr_result}}

            # ✅ Merge OCR results with template (OCR values override template defaults)
            merged_data = {}
            for page, ocr_fields in ocr_result.items():
                # Ensure ocr_fields is a dict
                if isinstance(ocr_fields, str):
                    try:
                        ocr_fields = json.loads(ocr_fields)
                    except json.JSONDecodeError:
                        ocr_fields = {"raw_text": ocr_fields}
                elif ocr_fields is None:
                    ocr_fields = {}

                # Get template for this page (use page_1 template for all pages for now)
                page_template = template.copy()

                # Merge: template fields + OCR extracted values
                merged_page_data = page_template.copy()
                merged_page_data.update(ocr_fields)
                merged_data[page] = merged_page_data

            # ✅ If no pages in OCR result, use template
            if not merged_data:
                merged_data = {"page_1": template}

            document.extracted_data = merged_data

            # ✅ Build extracted_text for search
            text_lines = []
            for page, fields in merged_data.items():
                for field_name, field_value in fields.items():
                    if field_value:  # Only include non-empty fields
                        text_lines.append(f"{field_name}: {field_value}")

            document.extracted_text = "\n".join(text_lines)
            document.processed = True
            document.save()

            messages.success(request, "✅ Document processed successfully!")

        except Exception as e:
            logger.error(f"OCR processing failed for document {document.id}: {e}")
            document.error_message = str(e)
            document.save()
            messages.warning(
                request,
                f"⚠️ Document uploaded but OCR failed: {str(e)}. "
                "You can manually edit the extracted data.",
            )

        return redirect("core:document_detail", pk=document.id)

    form = DocumentUploadForm()
    return render(request, "upload.html", {"form": form})


@login_required
def document_detail(request, pk):
    """Show details of a single document."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        # Update extracted data inline
        new_data = {
            k: v[0] if isinstance(v, list) else v
            for k, v in request.POST.items()
            if k != "csrfmiddlewaretoken"
        }
        document.extracted_data.update(new_data)
        document.save()
        return redirect("core:document_detail", pk=document.id)

    return render(request, "document_detail.html", {"document": document})


@login_required
@csrf_exempt
def chat_api(request):
    """Handle chat messages and base64 document uploads with template integration."""
    if request.method != "POST":
        return JsonResponse({"error": "Invalid method"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # ------------------------------
    # Case 1: Chat message search
    # ------------------------------
    user_message = data.get("message")
    if user_message:
        docs = Document.objects.filter(user=request.user, processed=True)
        response_text = "I couldn't find any matching info in your documents."
        threshold = 70  # similarity % for fuzzy match
        best_match_score = 0
        best_match_response = response_text

        for doc in docs:
            if doc.extracted_data:
                for page, fields in doc.extracted_data.items():
                    # Ensure fields is a dict
                    if isinstance(fields, str):
                        try:
                            fields = json.loads(fields)
                        except json.JSONDecodeError:
                            fields = {"raw_text": fields}
                    elif fields is None:
                        fields = {}

                    for key, value in fields.items():
                        if not value:  # Skip empty values
                            continue

                        key_str = str(key)
                        value_str = str(value)

                        # Fuzzy matching for typo-tolerance
                        key_ratio = fuzz.partial_ratio(
                            user_message.lower(), key_str.lower()
                        )
                        value_ratio = fuzz.partial_ratio(
                            user_message.lower(), value_str.lower()
                        )

                        current_score = max(key_ratio, value_ratio)

                        if (
                            current_score >= threshold
                            and current_score > best_match_score
                        ):
                            best_match_score = current_score
                            best_match_response = (
                                f"📄 Found in {doc.get_doc_type_display()}:\n"
                                f"**{key}**: {value}"
                            )

                            # If we have a very high match, return immediately
                            if current_score >= 90:
                                return JsonResponse({"response": best_match_response})

        if best_match_score >= threshold:
            return JsonResponse({"response": best_match_response})
        else:
            return JsonResponse({"response": response_text})

    # ------------------------------
    # Case 2: Base64 file upload
    # ------------------------------
    file_base64 = data.get("fileBase64")
    file_type = data.get("fileType")
    if file_base64 and file_type:
        tmp_file_path = None
        try:
            # Decode and save temporary file
            decoded_file = base64.b64decode(file_base64)
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_file.write(decoded_file)
            tmp_file.flush()
            tmp_file_path = tmp_file.name
            tmp_file.close()

            # ✅ Get template for this document type
            template = DOCUMENT_FIELD_TEMPLATES.get(
                file_type, DOCUMENT_FIELD_TEMPLATES["other_document"]
            ).copy()

            # Process OCR
            ocr_result = process_document_file(tmp_file_path, file_type)

            # ✅ Ensure OCR result is properly structured and merge with template
            if not ocr_result:
                ocr_result = {"page_1": {}}
            elif isinstance(ocr_result, str):
                ocr_result = {"page_1": {"raw_text": ocr_result}}

            merged_data = {}
            for page, ocr_fields in ocr_result.items():
                # Ensure ocr_fields is a dict
                if isinstance(ocr_fields, str):
                    try:
                        ocr_fields = json.loads(ocr_fields)
                    except json.JSONDecodeError:
                        ocr_fields = {"raw_text": ocr_fields}
                elif ocr_fields is None:
                    ocr_fields = {}

                # Merge template with OCR results
                page_template = template.copy()
                merged_page_data = page_template.copy()
                merged_page_data.update(ocr_fields)
                merged_data[page] = merged_page_data

            # ✅ If no pages in OCR result, use template
            if not merged_data:
                merged_data = {"page_1": template}

            # ✅ Build readable text response
            text_lines = ["📋 Extracted Data:"]
            for page, fields in merged_data.items():
                text_lines.append(f"\n📄 {page}:")
                for field_name, field_value in fields.items():
                    if field_value:  # Only show non-empty fields
                        text_lines.append(f"  • {field_name}: {field_value}")

            extracted_text = "\n".join(text_lines)

            return JsonResponse(
                {
                    "text": extracted_text,
                    "structured_data": merged_data,
                    "message": "✅ Document processed successfully with template fields.",
                }
            )

        except Exception as e:
            logger.error(f"OCR failed on base64 upload: {e}")
            return JsonResponse(
                {"error": f"OCR processing failed: {str(e)}"}, status=500
            )
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    return JsonResponse({"error": "No message or document data provided"}, status=400)
