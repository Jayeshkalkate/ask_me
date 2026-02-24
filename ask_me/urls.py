from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path("", include(("core.urls", "core"), namespace="core")),
    path("home/", views.homepage, name="homepage"),
    path("account/", include(("account.urls", "account"), namespace="account")),
    path("account/", include("django.contrib.auth.urls")),
    path("userprofile/", views.userprofile, name="userprofile"),
    path("aboutus/", views.aboutus, name="aboutus"),
    path("services/", views.services, name="services"),
    path("contactus/", views.contact, name="contactus"),
    path("privacy-policy/", views.privacy_policy, name="privacy_policy"),
    path("termsandconditions/", views.termsandconditions, name="termsandconditions"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
