# C:\chatbot\ask_me\core\views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import json, base64, tempfile, os, logging
from difflib import get_close_matches

from .forms import DocumentUploadForm
from .models import Document
from .ocr_utils import process_document_file

# Set up logging
logger = logging.getLogger(__name__)


@login_required
def homepage(request):
    """Render the home page."""
    return render(request, "index.html")


# ✅ Step 2: Edit Document View
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
    """Handle document upload and OCR extraction."""
    if request.method == "POST" and request.FILES.get("file"):
        doc_file = request.FILES["file"]
        doc_type = request.POST.get("doc_type", None)

        document = Document.objects.create(
            user=request.user, file=doc_file, doc_type=doc_type
        )

        try:
            extracted_data = process_document_file(document.file.path, doc_type)
            document.extracted_data = extracted_data
            document.extracted_text = "\n".join(
                f"{page}: {data}" for page, data in extracted_data.items()
            )
            document.processed = True
            document.save()
        except Exception as e:
            logger.warning(f"OCR processing failed for document {document.id}: {e}")
            document.error_message = str(e)
            document.save()

        return redirect("core:document_detail", pk=document.id)

    form = DocumentUploadForm()
    return render(request, "upload.html", {"form": form})


@login_required
def document_detail(request, pk):
    """Show details of a single document."""
    document = get_object_or_404(Document, pk=pk, user=request.user)

    if request.method == "POST":
        # Update extracted data inline (optional future use)
        new_data = {
            k: v[0] if isinstance(v, list) else v
            for k, v in request.POST.items()
            if k != "csrfmiddlewaretoken"
        }
        document.extracted_data.update(new_data)
        document.save()
        return redirect("core:document_detail", pk=document.id)

    return render(request, "document_detail.html", {"document": document})


from rapidfuzz import fuzz
import json, base64, tempfile, os
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Document
from .ocr_utils import process_document_file
import logging

logger = logging.getLogger(__name__)


@login_required
@csrf_exempt
def chat_api(request):
    """Handle chat messages and base64 document uploads with typo-tolerant search."""
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
                        key_str = str(key)
                        value_str = str(value)

                        # Fuzzy matching for typo-tolerance
                        key_ratio = fuzz.partial_ratio(
                            user_message.lower(), key_str.lower()
                        )
                        value_ratio = fuzz.partial_ratio(
                            user_message.lower(), value_str.lower()
                        )

                        if key_ratio >= threshold or value_ratio >= threshold:
                            response_text = f"Found in {doc.get_doc_type_display()} - {key}: {value}"
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

        return JsonResponse({"response": response_text})

    # ------------------------------
    # Case 2: Base64 file upload
    # ------------------------------
    file_base64 = data.get("fileBase64")
    file_type = data.get("fileType")
    if file_base64 and file_type:
        tmp_file_path = None
        try:
            decoded_file = base64.b64decode(file_base64)
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            tmp_file.write(decoded_file)
            tmp_file.flush()
            tmp_file_path = tmp_file.name
            tmp_file.close()

            ocr_result = process_document_file(tmp_file_path, file_type)

            # Ensure all pages are dicts
            for page, fields in ocr_result.items():
                if isinstance(fields, str):
                    try:
                        ocr_result[page] = json.loads(fields)
                    except json.JSONDecodeError:
                        ocr_result[page] = {"raw_text": fields}
                elif fields is None:
                    ocr_result[page] = {}

            extracted_text = "\n".join(
                f"{page}: {fields}" for page, fields in ocr_result.items()
            )
            return JsonResponse({"text": extracted_text})

        except Exception as e:
            logger.warning(f"OCR failed on base64 upload: {e}")
            return JsonResponse({"error": str(e)}, status=500)
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    return JsonResponse({"error": "No message or document data provided"}, status=400)
