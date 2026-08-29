import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from .forms import UserRegistrationForm, CustomAuthenticationForm
from .models import Profile

logger = logging.getLogger(__name__)


@user_passes_test(lambda u: u.is_superuser)
def admin_user_list(request):
    users = User.objects.select_related('profile').all()
    return render(request, 'admin_user_list.html', {'users': users})


@user_passes_test(lambda u: u.is_superuser)
def delete_user(request, user_id):
    user_to_delete = get_object_or_404(User, id=user_id)
    if user_to_delete == request.user:
        messages.error(request, 'You cannot delete your own account.')
        return redirect('account:admin_user_list')
    logger.info(f'Deleting user: {user_to_delete.username} (ID: {user_to_delete.id})')
    user_to_delete.delete()
    messages.success(request, f'User {user_to_delete.username} deleted successfully.')
    return redirect('account:admin_user_list')


def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('core:index')
        else:
            for error in form.non_field_errors():
                messages.error(request, error)
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = UserRegistrationForm()
    return render(request, 'register.html', {'form': form})


def user_login(request):
    if request.user.is_authenticated:
        return redirect('core:index')
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                logger.info(f'User {username} logged in successfully.')
                return redirect('core:index')
        else:
            logger.warning(f'Failed login attempt for username: {request.POST.get("username")}')
            messages.error(request, 'Invalid username or password.')
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login.html', {'form': form})


def user_logout(request):
    logout(request)
    return redirect('account:login')


@login_required(login_url='account:login')
def home(request):
    profile = get_object_or_404(Profile, user=request.user)
    return render(request, 'index.html', {'profile': profile})