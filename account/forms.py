from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError


class UserRegistrationForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, label='Password')
    confirm_password = forms.CharField(widget=forms.PasswordInput, label='Confirm Password')
    full_name = forms.CharField(max_length=100, label='Full Name')
    phone_number = forms.CharField(max_length=15, label='Phone Number')
    address = forms.CharField(widget=forms.Textarea, label='Address')
    city = forms.CharField(max_length=100, label='City')

    class Meta:
        model = User
        fields = ('username', 'email', 'password')

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError('Username already exists.')
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError('Email already registered.')
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            raise ValidationError('Passwords do not match.')
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])

        # Split full name into first and last name
        full_name = self.cleaned_data['full_name']
        parts = full_name.split(' ', 1)
        user.first_name = parts[0]
        user.last_name = parts[1] if len(parts) > 1 else ''

        if commit:
            user.save()
            # Create the associated profile
            Profile.objects.create(
                user=user,
                phone_number=self.cleaned_data['phone_number'],
                address=self.cleaned_data['address'],
                city=self.cleaned_data['city']
            )
        return user


class CustomAuthenticationForm(AuthenticationForm):
    """Optional: add custom validation or styling."""
    pass
