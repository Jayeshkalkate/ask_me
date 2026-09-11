from django.apps import AppConfig

class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        # Text extraction uses Google Gemini's free API tier (see
        # core/ai_extract.py) via GEMINI_API_KEY in settings.py.
        # No startup configuration needed here.
        pass