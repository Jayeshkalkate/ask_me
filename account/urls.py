from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'account'

urlpatterns = [
    # Custom authentication
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('home/', views.home, name='home'),

    # Admin user management
    path('admin-user-list/', views.admin_user_list, name='admin_user_list'),
    path('admin/delete-user/<int:user_id>/', views.delete_user, name='delete_user'),

    # Password reset
    path('password-reset/',
         auth_views.PasswordResetView.as_view(
             template_name='password_reset_form.html',
             email_template_name='password_reset_email.html',
             subject_template_name='password_reset_subject.txt'
         ),
         name='password_reset'),
    path('password-reset/done/',
         auth_views.PasswordResetDoneView.as_view(
             template_name='password_reset_done.html'
         ),
         name='password_reset_done'),
    path('password-reset/<uidb64>/<token>/',
         auth_views.PasswordResetConfirmView.as_view(
             template_name='password_reset_confirm.html'
         ),
         name='password_reset_confirm'),
    path('password-reset/complete/',
         auth_views.PasswordResetCompleteView.as_view(
             template_name='password_reset_complete.html'
         ),
         name='password_reset_complete'),
]