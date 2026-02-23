from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.homepage, name="homepage"),
    path("", include("core.urls", namespace="core")),
    path("account/", include(("account.urls", "account"), namespace="account")),
    path("account/", include("django.contrib.auth.urls")),
    path("userprofile/", views.userprofile, name="userprofile"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 
    