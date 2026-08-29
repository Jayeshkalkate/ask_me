from django import forms
from django.core.exceptions import ValidationError
import json
import os

from .models import Document, convert_numpy


class DocumentUploadForm(forms.ModelForm):
    """
    Form for uploading a document with type selection.
    Includes client‑side and server‑side validation.
    """

    doc_type = forms.ChoiceField(
        choices=Document.DOC_TYPES,
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Document Type",
        required=False,
        help_text="Select the document type (auto‑detected if not set).",
    )

    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={"class": "form-control"}),
        label="Upload Document",
        help_text="Supported formats: PDF, JPG, JPEG, PNG (max 10MB).",
    )

    class Meta:
        model = Document
        fields = ["file", "doc_type"]

    def clean_file(self):
        """Validate the uploaded file (size, extension, content)."""
        file = self.cleaned_data.get("file")
        if not file:
            raise ValidationError("Please select a file to upload.")

        # Additional size check (model validator already does this, but we add a
        # user‑friendly message here).
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size > max_size:
            raise ValidationError(f"File size must be under {max_size // (1024*1024)}MB.")

        # Check extension (model validator does it, but we can give immediate feedback).
        ext = os.path.splitext(file.name)[1].lower()
        allowed_extensions = [".pdf", ".jpg", ".jpeg", ".png"]
        if ext not in allowed_extensions:
            raise ValidationError(
                f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}."
            )

        return file

    def clean(self):
        """Cross‑field validation (if needed)."""
        cleaned_data = super().clean()
        # If doc_type is not provided, it will be auto‑detected in the view.
        # No additional checks needed here.
        return cleaned_data


class DocumentEditForm(forms.ModelForm):
    """
    Form for editing a document's structured JSON data.
    Converts user input to a Python dict and validates JSON format.
    """

    user_edited_data = forms.CharField(
        widget=forms.Textarea(
            attrs={
                "class": (
                    "w-full h-96 p-4 border border-gray-300 dark:border-gray-600 "
                    "rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 "
                    "font-mono text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 "
                    "transition-colors resize-vertical"
                ),
                "placeholder": "Edit your document data in JSON format...",
                "rows": 20,
                "spellcheck": "false",
            }
        ),
        label="Document Data (JSON)",
        required=False,
        help_text=(
            "Edit the JSON data below. Make sure to maintain valid JSON format. "
            "After saving, the original extracted data will be replaced with your edits."
        ),
    )

    class Meta:
        model = Document
        fields = ["user_edited_data"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre‑fill the textarea with the current display data as JSON.
        if self.instance and self.instance.pk:
            display_data = self.instance.display_data
            # If display_data is empty, show an empty JSON object.
            if not display_data:
                display_data = {}
            self.initial["user_edited_data"] = json.dumps(
                display_data, indent=2, ensure_ascii=False
            )

    def clean_user_edited_data(self):
        """Validate and parse the JSON input."""
        data = self.cleaned_data.get("user_edited_data")
        if data is None or data.strip() == "":
            # Allow empty JSON (user wants to clear data)
            return {}

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            # Provide a clear error message with line/column info.
            raise ValidationError(
                f"Invalid JSON format at line {e.lineno}, column {e.colno}: {e.msg}. "
                "Please check your input."
            )

        # Ensure the parsed data is a dict (or list) – we expect dict.
        if not isinstance(parsed, dict):
            raise ValidationError(
                "Data must be a JSON object (dictionary), not an array or primitive."
            )

        # Convert any NumPy types to native Python (safe for JSONField).
        try:
            parsed = convert_numpy(parsed)
        except Exception as e:
            raise ValidationError(f"Error processing data: {str(e)}")

        return parsed

    def save(self, commit=True):
        """
        Save the form – but we override to ensure we only update the
        user_edited_data field and clear extracted_data.
        The view already handles this via `document.update_user_data()`.
        This method is kept for completeness.
        """
        instance = super().save(commit=False)
        # The cleaned data is already a dict; assign it.
        instance.user_edited_data = self.cleaned_data["user_edited_data"]
        # Optionally clear extracted_data to indicate user edits override.
        instance.extracted_data = {}
        if commit:
            instance.save(update_fields=["user_edited_data", "extracted_data"])
        return instance
