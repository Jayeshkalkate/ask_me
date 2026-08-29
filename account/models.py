from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta


class Profile(models.Model):
    """
    Extended user profile (one-to-one with User).
    Stores additional info: phone, address, city, and last activity.
    """
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'          # use user.profile instead of user.profile
    )
    phone_number = models.CharField(max_length=15)
    address = models.TextField()
    city = models.CharField(max_length=100)
    last_activity = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['user__username']

    def __str__(self):
        return f"Profile of {self.user.username}"

    def is_online(self):
        """Return True if last activity was within the last 5 minutes."""
        return timezone.now() - self.last_activity < timedelta(minutes=5)
