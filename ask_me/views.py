from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def homepage(request):
    """Render chatbot homepage."""
    return render(request, "index.html")


@login_required(login_url="login")
def userprofile(request):
    return render(request, "userprofile.html")


def admin_user_list(request):
    return render(request, "admin_user_list.html")
