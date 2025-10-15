# C:\chatbot\ask_me\ask_me\views.py

from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages

from django.contrib.auth.models import User
from .forms import UserForm, ProfileForm
from .models import Profile

@login_required
def homepage(request):
    """Render chatbot homepage."""
    return render(request, "index.html")

@login_required(login_url="login")
def userprofile(request):
    return render(request, "userprofile.html")
