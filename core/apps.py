import os
import pytesseract
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):

        # 1️⃣ First check ENV variable (best practice)
        env_path = os.getenv("PYTESSERACT_CMD")
        if env_path:
            pytesseract.pytesseract.tesseract_cmd = env_path
            print(f"✅ Tesseract configured from ENV: {env_path}")
            return

        # 2️⃣ Windows fallback
        if os.name == "nt":
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                print("✅ Tesseract configured for Windows")
            else:
                print("⚠️ Windows Tesseract not found")
