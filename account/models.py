from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta


class Items(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15)
    address = models.TextField()
    city = models.CharField(max_length=100)
    last_activity = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.user.username

    def is_online(self):
        if self.last_activity:
            return timezone.now() - self.last_activity < timedelta(minutes=5)
        return False

    def __str__(self):
        return self.user.username
