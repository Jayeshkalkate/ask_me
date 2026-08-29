# core/urls.py
from django.urls import path
from . import views
from . import views_offline

app_name = "core"

urlpatterns = [
    path("", views.homepage, name="index"),
    path("upload/", views.upload_document, name="upload"),
    path("document/<int:pk>/", views.document_detail, name="document_detail"),
    path("document/<int:pk>/edit/", views.edit_document, name="edit_document"),
    path(
        "document/<int:pk>/reprocess/",
        views.reprocess_document,
        name="reprocess_document",
    ),
    path("document/<int:pk>/delete/", views.delete_document, name="delete_document"),
    path("documents/", views.document_library, name="document_library"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path(
        "documents/search/", views.search_document_field, name="search_document_field"
    ),
    
    # Offline API endpoints
    path('api/offline/upload/', views_offline.offline_upload, name='offline_upload'),
    path('api/offline/documents/', views_offline.offline_documents_api, name='offline_documents_api'),

    # ... existing patterns ...
    path('api/chat/', views.chat_api, name='chat_api'),
    path('api/base64-upload/', views.handle_base64_upload, name='base64_upload'),
    path('api/document-quality/<int:pk>/', views.analyze_document_quality, name='document_quality'),
    path('system-status/', views.system_status, name='system_status'),
]