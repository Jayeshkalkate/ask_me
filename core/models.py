from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.urls import reverse
from django.utils import timezone
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 🔹 NUMPY SAFE CONVERSION (recursive)
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
# 🔹 DOCUMENT FIELD TEMPLATES (for UI hints)
# -------------------------------------------------
DOCUMENT_FIELD_TEMPLATES = {
    # Identity Documents
    "aadhaar_card": {
        "Full Name": "",
        "Aadhaar Number": "",
        "Date of Birth": "",
        "Gender": "",
        "Address": "",
        "QR Code": "",
    },
    "pan_card": {
        "Full Name": "",
        "PAN Number": "",
        "Date of Birth": "",
        "Father's Name": "",
        "Date of Issue": "",
    },
    "voter_id_card": {
        "Full Name": "",
        "Voter ID Number": "",
        "Date of Birth": "",
        "Gender": "",
        "Father/Husband Name": "",
        "Assembly Constituency": "",
        "Address": "",
    },
    "passport": {
        "Full Name": "",
        "Passport Number": "",
        "Nationality": "",
        "Date of Birth": "",
        "Place of Birth": "",
        "Date of Issue": "",
        "Expiry Date": "",
        "File Number": "",
        "Country Code": "",
    },
    "driving_license": {
        "Full Name": "",
        "License Number": "",
        "Date of Birth": "",
        "Blood Group": "",
        "Address": "",
        "Date of Issue": "",
        "Valid Until": "",
        "Vehicle Class": "",
        "Transport Authority": "",
    },
    "vehicle_registration_certificate": {
        "Owner Name": "",
        "Registration Number": "",
        "Vehicle Type": "",
        "Registration Date": "",
        "Engine Number": "",
        "Chassis Number": "",
        "Vehicle Manufacturer": "",
        "Fuel Type": "",
        "Color": "",
        "Seating Capacity": "",
        "Valid Until": "",
    },
    
    # Civil & Residency Documents
    "domicile_certificate": {
        "Applicant Name": "",
        "Period of Residence": "",
        "Certificate Number": "",
        "Date of Issue": "",
        "Authority": "",
    },
    "nationality_certificate": {
        "Applicant Name": "",
        "Citizenship Declaration": "",
        "Outward Number": "",
        "Issuing Authority": "",
        "Date of Issue": "",
    },
    "birth_certificate": {
        "Full Name": "",
        "Date of Birth": "",
        "Place of Birth": "",
        "Mother's Name": "",
        "Father's Name": "",
        "Registration Number": "",
        "Date of Registration": "",
    },
    "marriage_certificate": {
        "Husband's Name": "",
        "Wife's Name": "",
        "Date of Marriage": "",
        "Place of Marriage": "",
        "Registration Number": "",
        "Date of Registration": "",
    },
    "death_certificate": {
        "Deceased Name": "",
        "Date of Death": "",
        "Place of Death": "",
        "Cause of Death": "",
        "Registration Number": "",
        "Date of Registration": "",
    },
    
    # Land & Property Documents
    "property_card": {
        "Survey Number": "",
        "CTS Number": "",
        "Plot Number": "",
        "Owner Name": "",
        "Land Area": "",
        "Village Name": "",
        "Cadastral Map Reference": "",
        "Tax Assessment Details": "",
    },
    "income_certificate": {
        "Applicant Name": "",
        "Annual Family Income": "",
        "Financial Year": "",
        "Certificate Number": "",
        "Authority": "",
        "Date of Issue": "",
    },
    "ration_card": {
        "Ration Card Number": "",
        "Household Head Name": "",
        "Family Member Count": "",
        "Income Category": "",
        "Card Type": "",
        "Address": "",
    },
    
    # Educational Documents
    "school_leaving_certificate": {
        "Student Name": "",
        "Date of Birth": "",
        "Place of Birth": "",
        "Religion": "",
        "Caste": "",
        "Date of Leaving": "",
        "School Name": "",
    },
    "ssc_marksheet": {
        "Student Name": "",
        "Seat Number": "",
        "Center Code": "",
        "Subject Marks": "",
        "Total Percentage": "",
        "Division/Grade": "",
        "Year": "",
    },
    "hsc_marksheet": {
        "Student Name": "",
        "Seat Number": "",
        "Center Code": "",
        "Subject Marks": "",
        "Total Percentage": "",
        "Division/Grade": "",
        "Year": "",
    },
    "degree_certificate": {
        "Student Name": "",
        "Degree Serial Number": "",
        "Passing Year": "",
        "Course Name": "",
        "Classification/Class": "",
        "University Name": "",
    },
    "board_passing_certificate": {
        "Student Name": "",
        "Passing Year": "",
        "Board/University Name": "",
        "Total Marks Obtained": "",
        "Result Status": "",
    },
    
    # Category & Welfare Certificates
    "caste_certificate": {
        "Applicant Name": "",
        "Caste Category": "",
        "Sub-Caste Name": "",
        "Authority Signature": "",
        "Outward Number": "",
        "Date of Issue": "",
    },
    "caste_validity_certificate": {
        "Applicant Name": "",
        "Scrutiny Committee Decision": "",
        "Validity Certificate Number": "",
        "Date of Issue": "",
        "Case Number": "",
    },
    "non_creamy_layer_certificate": {
        "Applicant Name": "",
        "Financial Year": "",
        "OBC Sub-Caste": "",
        "Validity Expiry Date": "",
        "Certificate Number": "",
    },
    "ews_certificate": {
        "Applicant Name": "",
        "Asset Valuation Details": "",
        "Annual Income Limit Verification": "",
        "Certificate Number": "",
        "Date of Issue": "",
    },
    
    # Other Documents
    "gst_certificate": {
        "GST Number": "",
        "Business Name": "",
        "Business Type": "",
        "Date of Registration": "",
        "State": "",
        "Address": "",
        "Status": "",
    },
    "invoice": {
        "Invoice Number": "",
        "Invoice Date": "",
        "Customer Name": "",
        "Customer GST": "",
        "Total Amount": "",
        "Tax Amount": "",
        "Items": "",
    },
    "other_document": {
        "Document Type": "",
        "Content": "",
        "Date": "",
        "Reference Number": "",
    },
}


# -------------------------------------------------
# 🔹 FILE VALIDATORS
# -------------------------------------------------
def validate_file_size(value):
    max_size = 10 * 1024 * 1024  # 10MB
    if value.size > max_size:
        raise ValidationError(f"File size must be under {max_size // (1024*1024)}MB.")


def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1].lower()
    valid_extensions = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    if ext not in valid_extensions:
        raise ValidationError(
            f"Unsupported file type. Allowed: {', '.join(valid_extensions)}."
        )


# -------------------------------------------------
# 🔹 CUSTOM MANAGER
# -------------------------------------------------
class DocumentManager(models.Manager):
    def get_display_data(self, document):
        """Return (display_data, is_user_edited) tuple."""
        if document.user_edited_data:
            return document.user_edited_data, True
        return document.extracted_data, False

    def processed(self):
        """Return only successfully processed documents."""
        return self.filter(processed=True, error_message__isnull=True)

    def failed(self):
        """Return documents that failed processing."""
        return self.filter(error_message__isnull=False)
    
    def by_doc_type(self, doc_type):
        """Return documents of a specific type."""
        return self.filter(doc_type=doc_type)
    
    def for_user(self, user):
        """Return documents for a specific user."""
        return self.filter(user=user)


# -------------------------------------------------
# 🔹 DOCUMENT MODEL
# -------------------------------------------------
class Document(models.Model):
    # Document type choices - Complete list of all document types
    DOC_TYPES = [
        # Identity Documents
        ("aadhaar_card", "Aadhaar Card"),
        ("pan_card", "PAN Card"),
        ("voter_id_card", "Voter ID Card"),
        ("passport", "Passport"),
        ("driving_license", "Driving License"),
        ("vehicle_registration_certificate", "Vehicle Registration Certificate"),
        
        # Civil & Residency Documents
        ("domicile_certificate", "Domicile Certificate"),
        ("nationality_certificate", "Nationality Certificate"),
        ("birth_certificate", "Birth Certificate"),
        ("marriage_certificate", "Marriage Certificate"),
        ("death_certificate", "Death Certificate"),
        
        # Land & Property Documents
        ("property_card", "Property Card"),
        ("income_certificate", "Income Certificate"),
        ("ration_card", "Ration Card"),
        
        # Educational Documents
        ("school_leaving_certificate", "School Leaving Certificate"),
        ("ssc_marksheet", "SSC Marksheet"),
        ("hsc_marksheet", "HSC Marksheet"),
        ("degree_certificate", "Degree Certificate"),
        ("board_passing_certificate", "Board Passing Certificate"),
        
        # Category & Welfare Certificates
        ("caste_certificate", "Caste Certificate"),
        ("caste_validity_certificate", "Caste Validity Certificate"),
        ("non_creamy_layer_certificate", "Non-Creamy Layer Certificate"),
        ("ews_certificate", "EWS Certificate"),
        
        # Other Documents
        ("gst_certificate", "GST Certificate"),
        ("invoice", "Invoice"),
        ("other_document", "Other Document"),
    ]

    # Relationships
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="documents",
        db_index=True,
    )

    # Document metadata
    doc_type = models.CharField(
        max_length=50,
        choices=DOC_TYPES,
        default="other_document",
        blank=True,
        null=True,
    )

    # File storage
    file = models.FileField(
        upload_to="documents/%Y/%m/%d/",
        validators=[validate_file_size, validate_file_extension],
        null=True,
        blank=True,
    )

    # Raw OCR text (cached)
    extracted_text = models.TextField(blank=True, null=True)

    # Structured data (JSON)
    extracted_data = models.JSONField(default=dict, blank=True)
    user_edited_data = models.JSONField(default=dict, blank=True)
    ai_extracted_json = models.JSONField(default=dict, blank=True)  # cache for AI extraction

    # Processing status
    processed = models.BooleanField(default=False, db_index=True)
    processed_at = models.DateTimeField(blank=True, null=True)
    error_message = models.TextField(blank=True, null=True)
    quality_score = models.FloatField(blank=True, null=True)

    # File information
    original_filename = models.CharField(max_length=255, blank=True, null=True)
    file_hash = models.CharField(max_length=64, blank=True, null=True)

    # Sync status (for offline mode)
    synced = models.BooleanField(default=False, db_index=True)
    synced_at = models.DateTimeField(blank=True, null=True)
    offline_id = models.CharField(max_length=50, blank=True, null=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    uploaded_at = models.DateTimeField(auto_now=True)

    # Custom manager
    objects = DocumentManager()

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["doc_type", "processed"]),
            models.Index(fields=["user", "doc_type"]),
            models.Index(fields=["synced"]),
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        return f"{self.get_doc_type_display()} - {self.user.username} ({self.created_at.strftime('%Y-%m-%d')})"

    def get_absolute_url(self):
        """Return the URL to view this document."""
        return reverse("core:document_detail", kwargs={"pk": self.pk})

    # -------------------------------------------------
    # 🔹 DATA ACCESS HELPERS
    # -------------------------------------------------
    @property
    def display_data(self):
        """Return the data that should be displayed (user edited or extracted)."""
        return self.user_edited_data if self.user_edited_data else self.extracted_data

    @property
    def is_edited(self):
        """Return True if the user has edited this document's data."""
        return bool(self.user_edited_data)

    @property
    def file_size(self):
        """Return file size in bytes, or None if no file (or the file is
        missing/unreadable on disk — e.g. moved, deleted, or a media-root
        mismatch between environments). A single row's missing file should
        never take down the whole document list page."""
        if self.file and hasattr(self.file, "size"):
            try:
                return self.file.size
            except (FileNotFoundError, OSError):
                logger.warning(
                    "Document %s: file record exists but is missing on disk (%s)",
                    self.pk, getattr(self.file, "name", "?")
                )
                return None
        return None

    @property
    def file_extension(self):
        """Return file extension (lowercase) or empty string."""
        if self.file:
            return os.path.splitext(self.file.name)[1].lower()
        return ""

    @property
    def file_name(self):
        """Return the file name."""
        if self.file:
            return os.path.basename(self.file.name)
        return None

    @property
    def doc_type_display(self):
        """Get the display name for the document type."""
        return dict(self.DOC_TYPES).get(self.doc_type, self.doc_type.replace("_", " ").title())

    @property
    def is_offline(self):
        """Check if this document was created offline."""
        return bool(self.offline_id)

    # -------------------------------------------------
    # 🔹 DOCUMENT TYPE HELPERS
    # -------------------------------------------------
    def get_field_template(self):
        """Get the field template for this document type."""
        return DOCUMENT_FIELD_TEMPLATES.get(self.doc_type, DOCUMENT_FIELD_TEMPLATES["other_document"])

    def get_required_fields(self):
        """Get required fields for this document type."""
        template = self.get_field_template()
        return [field for field in template.keys() if field]

    def has_required_fields(self):
        """Check if all required fields are present in the extracted data."""
        required = self.get_required_fields()
        data = self.display_data
        if not data:
            return False
        # Check if all required fields exist in the data
        for page_data in data.values():
            if isinstance(page_data, dict):
                for field in required:
                    if field not in page_data or not page_data[field]:
                        return False
        return True

    # -------------------------------------------------
    # 🔹 DATA UPDATE METHODS
    # -------------------------------------------------
    def update_user_data(self, new_data):
        """Replace extracted data with user edited data."""
        self.user_edited_data = convert_numpy(new_data)
        self.extracted_data = {}
        self.save(update_fields=["user_edited_data", "extracted_data"])

    def reset_to_extracted(self):
        """Revert to extracted data (remove user edits)."""
        if self.extracted_data:
            self.user_edited_data = {}
            self.save(update_fields=["user_edited_data"])

    def merge_user_data(self, new_data):
        """Merge new data with existing user data."""
        current = self.user_edited_data if self.user_edited_data else {}
        merged = {**current, **convert_numpy(new_data)}
        self.user_edited_data = merged
        self.save(update_fields=["user_edited_data"])

    # -------------------------------------------------
    # 🔹 SYNC METHODS
    # -------------------------------------------------
    def mark_synced(self):
        """Mark this document as synced with the server."""
        self.synced = True
        self.synced_at = timezone.now()
        self.save(update_fields=["synced", "synced_at"])

    def mark_unsynced(self):
        """Mark this document as unsynced (created offline)."""
        self.synced = False
        self.synced_at = None
        self.save(update_fields=["synced", "synced_at"])

    # -------------------------------------------------
    # 🔹 AUTO SYNC ON SAVE
    # -------------------------------------------------
    def save(self, *args, **kwargs):
        """
        Automatically rebuild `extracted_text` from the active data
        (user_edited_data or extracted_data) for searchability.
        """
        source = self.user_edited_data if self.user_edited_data else self.extracted_data
        lines = []
        if isinstance(source, dict):
            for page_key, page_data in source.items():
                if isinstance(page_data, dict):
                    for field_key, field_value in page_data.items():
                        if field_key != "_metadata" and field_value:
                            if isinstance(field_value, str):
                                lines.append(f"{field_key}: {field_value}")
                            elif isinstance(field_value, (int, float)):
                                lines.append(f"{field_key}: {field_value}")
                            elif isinstance(field_value, dict):
                                # Handle nested dicts
                                for sub_key, sub_value in field_value.items():
                                    if sub_key != "_metadata" and sub_value:
                                        lines.append(f"{field_key}_{sub_key}: {sub_value}")
                            elif isinstance(field_value, list):
                                lines.append(f"{field_key}: {', '.join(str(item) for item in field_value)}")

        self.extracted_text = "\n".join(lines) if lines else ""

        # Ensure timestamps are set correctly
        if self.processed and not self.processed_at:
            self.processed_at = timezone.now()

        # Set original filename if not set
        if self.file and not self.original_filename:
            self.original_filename = os.path.basename(self.file.name)

        super().save(*args, **kwargs)

    # -------------------------------------------------
    # 🔹 CONVENIENCE METHODS
    # -------------------------------------------------
    def is_processing_successful(self):
        """Return True if the document was processed without errors."""
        return self.processed and not self.error_message

    def mark_as_failed(self, error_msg):
        """Mark the document as failed with an error message."""
        self.processed = False
        self.error_message = error_msg
        self.save(update_fields=["processed", "error_message"])

    def get_document_summary(self):
        """Get a summary of the document for display purposes."""
        summary = {
            "id": self.id,
            "type": self.doc_type_display,
            "user": self.user.username,
            "created": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "processed": self.processed,
            "edited": self.is_edited,
            "file_name": self.file_name,
            "file_size": self.file_size,
        }
        return summary

    def get_searchable_text(self):
        """Get all text content for search."""
        text_parts = []
        if self.extracted_text:
            text_parts.append(self.extracted_text)
        if self.user_edited_data:
            text_parts.append(str(self.user_edited_data))
        if self.extracted_data:
            text_parts.append(str(self.extracted_data))
        return " ".join(text_parts)
