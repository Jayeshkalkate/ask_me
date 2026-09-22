from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponse, HttpResponseNotFound
import logging
import os


def service_worker(request):
    """
    Serve static/sw.js at the site ROOT (/sw.js) instead of /static/sw.js.
    A service worker's default control scope is the folder it's served
    from, so registering it with scope: '/' (see static/js/pwa.js) only
    actually works if the browser fetches it from '/sw.js' — serving it
    from under /static/ would silently limit it to controlling only
    /static/* requests, breaking offline support for every app page.
    """
    sw_path = os.path.join(settings.BASE_DIR, "static", "sw.js")
    try:
        with open(sw_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return HttpResponseNotFound("sw.js not found")

    response = HttpResponse(content, content_type="application/javascript")
    # Belt-and-braces: makes the intended scope explicit even though
    # serving from '/' already implies it.
    response["Service-Worker-Allowed"] = "/"
    return response


def offline_page(request):
    """
    Render the offline fallback page at /offline.html.

    sw.js precaches this URL and serves it from cache whenever a
    navigation request fails with no network (see the 'fetch' handler
    in static/sw.js). No @login_required here on purpose: the service
    worker's install step fetches this on first visit, which can happen
    before the user has ever logged in, and it must also be servable
    entirely from cache while genuinely offline.
    """
    return render(request, "offline.html")


@login_required
def homepage(request):
    return render(request, "index.html")

def send_email_to_client(first_name, last_name, email, message):
    """
    Sends the contact-form message via EMAIL_BACKEND (Brevo's HTTPS API -
    see core/email_backends.py). Any failure here is caught rather than
    left to bubble up as a 500: a broken/unconfigured email provider
    should never crash the page for someone filling out the contact form.
    Returns True if the message was actually sent.
    """
    subject = "New Message from Client"
    full_message = f"Name: {first_name} {last_name}\nEmail: {email}\n\nMessage:\n{message}"
    from_address = settings.DEFAULT_FROM_EMAIL or settings.EMAIL_HOST_USER
    try:
        sent_count = send_mail(
            subject,
            full_message,
            from_address,
            ["jayeshkalkate432@gmail.com"],
            fail_silently=False,
        )
        return bool(sent_count)
    except Exception:
        logging.getLogger(__name__).exception("Failed to send contact-form email")
        return False

@login_required
def userprofile(request):
    return render(request, "userprofile.html")

def aboutus(request):
    return render(request, "aboutus.html")

def contact(request):
    if request.method == "POST":
        first_name = request.POST.get("first_name")
        last_name = request.POST.get("last_name")
        email = request.POST.get("email")
        message = request.POST.get("message")

        if first_name and last_name and email and message:
            if send_email_to_client(first_name, last_name, email, message):
                messages.success(request, "Email sent successfully!")
            else:
                messages.error(
                    request,
                    "Sorry, we couldn't send your message right now. Please try again later.",
                )
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