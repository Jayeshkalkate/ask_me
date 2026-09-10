from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import Profile

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'Profile'

class UserAdmin(BaseUserAdmin):
    inlines = [ProfileInline]
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')
    search_fields = ('username', 'email')
    list_filter = ('is_staff', 'is_superuser', 'is_active')

# Re-register User with custom admin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

# Register Profile separately for better management
@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'city', 'last_activity', 'is_online')
    list_filter = ('city',)
    search_fields = ('user__username', 'phone_number', 'address', 'city')
    readonly_fields = ('last_activity',)
    fields = ('user', 'phone_number', 'address', 'city', 'last_activity')

    def is_online(self, obj):
        return obj.is_online()
    is_online.boolean = True
    is_online.short_description = 'Online'
