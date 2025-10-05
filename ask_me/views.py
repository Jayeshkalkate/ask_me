# C:\chatbot\ask_me\ask_me\views.py

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .forms import DocumentUploadForm
from .models import Document
from ..core.ocr_utils import process_document_file

from django.contrib.auth.models import User
from .forms import UserForm, ProfileForm
from .models import Profile

@login_required
def homepage(request):
    """Render chatbot homepage."""
    return render(request, "index.html")


@login_required
def user_profile(request):
    """View and edit user profile."""
    if request.method == "POST":
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, "Profile updated successfully!")
            return redirect("ask_me:profile")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)

    context = {"user_form": user_form, "profile_form": profile_form}
    return render(request, "ask_me/profile.html", context)
