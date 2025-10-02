from django.contrib import admin
from django.contrib.auth.models import User
from .models import Items

# Register Items model separately (no inlines)
class ItemsAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'address', 'city')
    search_fields = ('user__username', 'phone_number', 'city')

admin.site.register(Items, ItemsAdmin)

# Extend the default User admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

class UserAdmin(BaseUserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')
    search_fields = ('username', 'email')

# Unregister the original User admin and register the custom one
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
