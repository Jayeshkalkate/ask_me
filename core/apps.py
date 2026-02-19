from django.apps import AppConfig
from django.apps import AppConfig
import pytesseract
import os

class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        if not os.path.exists(tesseract_path):
            raise RuntimeError(f"Tesseract not found at {tesseract_path}")

        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"✅ Tesseract configured at startup: {tesseract_path}")
