from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Tesseract is configured in settings.py and ocr_utils.py
        # No need to duplicate here
        pass