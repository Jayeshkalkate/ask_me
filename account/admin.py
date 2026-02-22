from django.contrib import admin
from django.contrib.auth.models import User
from .models import Items
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin


class ItemsAdmin(admin.ModelAdmin):
    list_display = ("user", "phone_number", "address", "city")
    search_fields = ("user__username", "phone_number", "city")


admin.site.register(Items, ItemsAdmin)


class UserAdmin(BaseUserAdmin):
    list_display = ("username", "email", "first_name", "last_name", "is_staff")
    search_fields = ("username", "email")


admin.site.unregister(User)
admin.site.register(User, UserAdmin)
