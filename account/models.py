from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta

class Profile(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='profile'
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
        return timezone.now() - self.last_activity < timedelta(minutes=5)

    def update_activity(self):
        """Helper to update last_activity without fetching full object."""
        self.last_activity = timezone.now()
        self.save(update_fields=["last_activity"])
