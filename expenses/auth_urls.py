"""
URL configuration for authentication-related endpoints.
"""

from django.urls import path
from expenses.user_views import UserProfileView

# Define app namespace
app_name = 'auth'

# Define the URL patterns
urlpatterns = [
    path('', UserProfileView.as_view(), name='user-profile'),
]
