# C:\chatbot\ask_me\ask_me\urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core.urls", namespace="core")),  # core app URLs with namespace
    path(
        "account/", include(("account.urls", "account"), namespace="account")
    ),  # account app with namespace
    path(
        "account/", include("django.contrib.auth.urls")
    ),  # ✅ add login/logout/password reset URLs
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
