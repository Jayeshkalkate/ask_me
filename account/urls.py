# C:\chatbot\ask_me\account\urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.user_login, name="login"),  # Root path goes to login
    path("register/", views.register, name="register"),
    path("login/", views.user_login, name="login"),
    path("logout/", views.user_logout, name="logout"),
    path("admin-user-list/", views.admin_user_list, name="admin_user_list"),
    path("admin/delete-user/<int:user_id>/", views.delete_user, name="delete_user"),
]
