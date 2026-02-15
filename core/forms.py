# C:\chatbot\ask_me\core\forms.py

from django import forms
from .models import Document
import json
from .models import convert_numpy

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

class DocumentEditForm(forms.ModelForm):
    """
    Enhanced form for editing document fields with better UX.
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
        help_text="Edit the JSON data below. Make sure to maintain valid JSON format. After saving, the original extracted data will be replaced with your edits.",
    )

    class Meta:
        model = Document
        fields = ["user_edited_data"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance:
            # Always show the current display data (user_edited_data prioritized)
            display_data = self.instance.display_data
            self.initial["user_edited_data"] = json.dumps(
                display_data, indent=2, ensure_ascii=False
            )

    def clean_user_edited_data(self):
        data = self.cleaned_data.get("user_edited_data")
        if not data or data.strip() == "":
            return {}
        try:
            parsed_data = json.loads(data)
            # Convert numpy types to native Python types
            return convert_numpy(parsed_data)
        except json.JSONDecodeError as e:
            raise forms.ValidationError(
                f"Invalid JSON format: {str(e)}. Please check your input."
            )
