from datetime import datetime
from django.utils import timezone


class ActiveUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            user_items = getattr(request.user, "items", None)
            if user_items:
                user_items.last_activity = timezone.now()
                user_items.save(update_fields=["last_activity"])
        return self.get_response(request)
