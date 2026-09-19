from datetime import timedelta
from django.utils import timezone

# Profile.is_online() only cares about a 5-minute window, so writing to the
# DB on literally every request (as this used to do) is unnecessary load -
# one page can fire off a dozen requests (assets, API calls, chat polling)
# in a few seconds. Only touch the DB once per this interval per user.
ACTIVITY_UPDATE_INTERVAL = timedelta(minutes=1)


class ActiveUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.user.is_authenticated:
            profile = getattr(request.user, "profile", None)
            if profile:
                now = timezone.now()
                if now - profile.last_activity >= ACTIVITY_UPDATE_INTERVAL:
                    profile.last_activity = now
                    profile.save(update_fields=["last_activity"])
        return self.get_response(request)
