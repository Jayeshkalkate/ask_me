from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages


@login_required
def homepage(request):
    return render(request, "index.html")


def send_email_to_client(first_name, last_name, email, message):
    subject = "New Message from Client"
    full_message = (
        f"Name: {first_name} {last_name}\nEmail: {email}\n\nMessage:\n{message}"
    )
    send_mail(
        subject, full_message, settings.EMAIL_HOST_USER, ["jayeshkalkate432@gmail.com"]
    )

def userprofile(request):
    return render(request, "userprofile.html")


def admin_user_list(request):
    return render(request, "admin_user_list.html")


def aboutus(request):
    return render(request, "aboutus.html")


def contact(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        email = request.POST.get("email")
        message = request.POST.get("message")

        if first_name and last_name and email and message:
            send_email_to_client(first_name, last_name, email, message)
            messages.success(request, "Email sent successfully!")
        else:
            messages.error(request, "Please fill in all fields.")

        return redirect("contactus")

    return render(request, "contact.html")


def services(request):
    return render(request, "services.html")

def privacy_policy(request):
    return render(request, "privacy_policy.html")

def termsandconditions(request):
    return render(request, "termsandconditions.html")
