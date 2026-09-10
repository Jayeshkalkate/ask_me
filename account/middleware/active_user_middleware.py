from datetime import datetime
from django.utils import timezone

class ActiveUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            profile = getattr(request.user, "profile", None)
            if profile:
                profile.last_activity = timezone.now()
                profile.save(update_fields=["last_activity"])
        return self.get_response(request)
