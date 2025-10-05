# C:\chatbot\ask_me\core\urls.py

from django.urls import path
from . import views

app_name = "core"

urlpatterns = [
    path("", views.homepage, name="index"),
    path("upload/", views.upload_document, name="upload"),
    path("document/<int:pk>/", views.document_detail, name="document_detail"),
    # ✅ Step 3: Add edit page URL (matches the button + view name)
    path("document/<int:pk>/edit/", views.edit_document, name="edit_document"),
    path("api/chat/", views.chat_api, name="chat_api"),
]
