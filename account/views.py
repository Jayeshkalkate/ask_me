from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from .models import Items
import logging

logger = logging.getLogger(__name__)


# ----------------- Admin Views -----------------
@user_passes_test(lambda u: u.is_superuser)
def admin_user_list(request):
    users = User.objects.select_related("items").all()
    return render(request, "admin_user_list.html", {"users": users})


@user_passes_test(lambda u: u.is_superuser)
def delete_user(request, user_id):
    user_to_delete = get_object_or_404(User, id=user_id)

    if user_to_delete == request.user:
        messages.error(request, "You cannot delete your own account.")
    else:
        # Log the deletion
        items = getattr(user_to_delete, "items", None)
        logger.info(f"Deleting user: {user_to_delete.username}, Items: {items}")

        # Delete the user
        user_to_delete.delete()
        messages.success(
            request, f"User {user_to_delete.username} deleted successfully."
        )

    return redirect("account:admin_user_list")


# ----------------- Authentication Views -----------------
def register(request):
    if request.method == "POST":
        full_name = request.POST.get("full_name")
        email = request.POST.get("email")
        phone_number = request.POST.get("phone_number")
        address = request.POST.get("address")
        city = request.POST.get("city")
        username = request.POST.get("username")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
        elif User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered")
        else:
            user = User.objects.create_user(
                username=username, email=email, password=password
            )
            first_name, *last_name = full_name.split(" ", 1)
            user.first_name = first_name
            user.last_name = last_name[0] if last_name else ""
            user.save()

            # Save additional info to Items model
            Items.objects.create(
                user=user, phone_number=phone_number, address=address, city=city
            )

            messages.success(request, "Account has been created successfully")
            login(request, user)  # Automatically log in after registration
            return redirect("core:index")  # Fixed: redirect using namespace

    return render(request, "register.html")


def user_login(request):
    # If user is already logged in, redirect to the homepage
    if request.user.is_authenticated:
        return redirect("core:index")  # Fixed

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        logger.info(f"Attempting to authenticate user: {username}")
        print(f"Attempting to authenticate user: {username}")  # Debug

        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Successful login
            logger.info(f"User {username} authenticated successfully.")
            print(f"User {username} authenticated successfully.")  # Debug
            login(request, user)
            return redirect("core:index")  # Fixed
        else:
            # Failed login attempt
            logger.warning(f"Failed login attempt for username: {username}")
            print(f"Failed login attempt for username: {username}")  # Debug
            messages.error(request, "Invalid Credentials")

    return render(request, "login.html")


def user_logout(request):
    logout(request)
    return redirect("account:login")  # Redirect to login page after logout


# ----------------- User Dashboard/Home -----------------
@login_required
def home(request):
    items = Items.objects.filter(
        user=request.user
    )  # Show only the logged-in user's items
    return render(request, "index.html", {"items": items})
