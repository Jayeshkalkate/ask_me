# C:\chatbot\ask_me\core\forms.py

from django import forms
from .models import Document


class DocumentUploadForm(forms.ModelForm):
    """Form for uploading documents with type selection."""

    doc_type = forms.ChoiceField(
        choices=Document.DOC_TYPES,
        widget=forms.Select(attrs={"class": "form-select"}),
        label="Document Type",
        required=False,
    )

    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={"class": "form-control"}),
        label="Upload Document",
    )

    class Meta:
        model = Document
        fields = ["file", "doc_type"]
