from django.db import models
from django.contrib.auth.models import User
import numpy as np
import json
import os
from django.core.exceptions import ValidationError


def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types for JSONField."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


# ✅ Document field templates for automatic extracted_data initialization
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
    "other_document": {
        "Document Type": "",
        "Content": "",
    },
}


class DocumentManager(models.Manager):
    def get_display_data(self, document):
        """Get data for display - prioritize user_edited_data"""
        if document.user_edited_data:
            return document.user_edited_data, True
        return document.extracted_data, False


# 🔒 File Validators

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

# Document Model

class Document(models.Model):
    """
    Model to store uploaded documents with OCR data and dynamic extracted fields.
    """

    DOC_TYPES = [
        ("aadhaar_card", "Aadhaar Card"),
        ("pan_card", "PAN Card"),
        ("driving_license", "Driving License"),
        ("passport", "Passport"),
        ("other_document", "Other Document"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="documents")
    # file = models.FileField(upload_to="documents/")
    file = models.FileField(
        upload_to="documents/", validators=[validate_file_size, validate_file_extension]
        )

    doc_type = models.CharField(
        max_length=50,
        choices=DOC_TYPES,
        default="other_document",
        blank=True,
        null=True,
    )

    # Raw OCR text
    extracted_text = models.TextField(blank=True, null=True)

    # Structured extracted data (from OCR) - will be deleted when user edits
    extracted_data = models.JSONField(default=dict, blank=True)

    # User-edited structured data - PRIMARY SOURCE after editing
    user_edited_data = models.JSONField(default=dict, blank=True)

    processed = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, null=True)
    quality_score = models.FloatField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now=True)

    objects = DocumentManager()

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        if self.doc_type:
            return f"{self.get_doc_type_display()} - {self.user.username} ({self.created_at.strftime('%Y-%m-%d')})"
        return f"{self.user.username} - {self.file.name}"

    def get_display_data(self):
        """Get data for display - prioritize user_edited_data"""
        return self.objects.get_display_data(self)

    def save(self, *args, **kwargs):
        """Override save to keep extracted_text in sync with user_edited_data"""
        text_lines = []

        # Priority: user_edited_data > extracted_data
        source_data = (
            self.user_edited_data if self.user_edited_data else self.extracted_data
        )

        for page_key, page_data in source_data.items():
            if isinstance(page_data, dict):
                for field_key, field_value in page_data.items():
                    if field_key != "_metadata" and field_value:
                        text_lines.append(f"{field_key}: {field_value}")

        # Update extracted_text only if there's content
        if text_lines:
            self.extracted_text = "\n".join(text_lines)
        else:
            self.extracted_text = ""

        super().save(*args, **kwargs)

    def update_user_data(self, new_data):
        """Update user_edited_data and clean up extracted_data"""
        self.user_edited_data = convert_numpy(new_data)

        # Clear extracted_data since user has edited the data
        self.extracted_data = {}

        # Rebuild extracted_text from user_edited_data
        self.save()

    @property
    def is_edited(self):
        """Check if document has user edits."""
        return bool(self.user_edited_data)

    @property
    def display_data(self):
        """Property to get display data easily"""
        return self.user_edited_data if self.user_edited_data else self.extracted_data
