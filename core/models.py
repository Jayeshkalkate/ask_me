from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
import numpy as np
import os


# -------------------------------------------------
# 🔹 NUMPY SAFE CONVERSION
# -------------------------------------------------


def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types for JSONField."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


# -------------------------------------------------
# 🔹 DOCUMENT FIELD TEMPLATES
# -------------------------------------------------

DOCUMENT_FIELD_TEMPLATES = {
    "aadhaar_card": {
        "Full Name": "",
        "Aadhaar Number": "",
        "Date of Birth": "",
        "Gender": "",
        "Address": "",
    },
    "pan_card": {
        "Full Name": "",
        "PAN Number": "",
        "Date of Birth": "",
        "Father's Name": "",
    },
    "driving_license": {
        "Full Name": "",
        "License Number": "",
        "Date of Birth": "",
        "Address": "",
        "Valid Until": "",
    },
    "passport": {
        "Full Name": "",
        "Passport Number": "",
        "Nationality": "",
        "Date of Birth": "",
        "Expiry Date": "",
    },
    "other_document": {
        "Document Type": "",
        "Content": "",
    },
}


# -------------------------------------------------
# 🔹 FILE VALIDATORS
# -------------------------------------------------


def validate_file_size(value):
    max_size = 10 * 1024 * 1024  # 10MB
    if value.size > max_size:
        raise ValidationError("File size must be under 10MB.")


def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]
    valid_extensions = [".pdf", ".jpg", ".jpeg", ".png"]
    if ext.lower() not in valid_extensions:
        raise ValidationError(
            "Unsupported file type. Allowed types: PDF, JPG, JPEG, PNG."
        )


# -------------------------------------------------
# 🔹 CUSTOM MANAGER
# -------------------------------------------------


class DocumentManager(models.Manager):
    def get_display_data(self, document):
        """Prioritize user edited data over extracted data"""
        if document.user_edited_data:
            return document.user_edited_data, True
        return document.extracted_data, False


# -------------------------------------------------
# 🔹 DOCUMENT MODEL
# -------------------------------------------------


class Document(models.Model):

    DOC_TYPES = [
        ("aadhaar_card", "Aadhaar Card"),
        ("pan_card", "PAN Card"),
        ("driving_license", "Driving License"),
        ("passport", "Passport"),
        ("other_document", "Other Document"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")

    doc_type = models.CharField(
        max_length=50,
        choices=DOC_TYPES,
        default="other_document",
        blank=True,
        null=True,
    )

    # -------------------------------------------------
    # 🔥 FILE STORAGE
    # -------------------------------------------------

    file = models.FileField(
        upload_to="documents/",
        validators=[validate_file_size, validate_file_extension],
        null=True,
        blank=True,
    )

    # -------------------------------------------------
    # 🔹 OCR + AI DATA STORAGE (CACHED)
    # -------------------------------------------------

    # Raw OCR text (or reconstructed text)
    extracted_text = models.TextField(blank=True, null=True)

    # Structured data from OCR parsing
    extracted_data = models.JSONField(default=dict, blank=True)

    # User edited structured data (highest priority)
    user_edited_data = models.JSONField(default=dict, blank=True)

    # ✅ AI structured extraction cache (VERY IMPORTANT)
    ai_extracted_json = models.JSONField(default=dict, blank=True)

    # -------------------------------------------------
    # 🔹 PROCESSING STATUS
    # -------------------------------------------------

    processed = models.BooleanField(default=False)

    # ✅ When OCR + AI completed
    processed_at = models.DateTimeField(blank=True, null=True)

    error_message = models.TextField(blank=True, null=True)
    quality_score = models.FloatField(blank=True, null=True)

    # -------------------------------------------------
    # 🔹 TIMESTAMPS
    # -------------------------------------------------

    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now=True)

    objects = DocumentManager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.get_doc_type_display()} - {self.user.username} ({self.created_at.strftime('%Y-%m-%d')})"

    # -------------------------------------------------
    # 🔹 DISPLAY DATA LOGIC
    # -------------------------------------------------

    def get_display_data(self):
        return self.objects.get_display_data(self)

    @property
    def display_data(self):
        return self.user_edited_data if self.user_edited_data else self.extracted_data

    @property
    def is_edited(self):
        return bool(self.user_edited_data)

    # -------------------------------------------------
    # 🔹 AUTO SYNC TEXT
    # -------------------------------------------------

    def save(self, *args, **kwargs):
        """
        Keep extracted_text synced with
        user edited OR extracted structured data.
        """

        text_lines = []

        source_data = (
            self.user_edited_data if self.user_edited_data else self.extracted_data
        )

        if isinstance(source_data, dict):
            for page_key, page_data in source_data.items():
                if isinstance(page_data, dict):
                    for field_key, field_value in page_data.items():
                        if field_key != "_metadata" and field_value:
                            text_lines.append(f"{field_key}: {field_value}")

        self.extracted_text = "\n".join(text_lines) if text_lines else ""

        super().save(*args, **kwargs)

    # -------------------------------------------------
    # 🔹 UPDATE USER DATA
    # -------------------------------------------------

    def update_user_data(self, new_data):
        self.user_edited_data = convert_numpy(new_data)
        self.extracted_data = {}
        self.save()
