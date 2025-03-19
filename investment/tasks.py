# investment/tasks.py
from celery import shared_task
from .models import UserProfile
from .services import calculate_and_update_disposable_income  # Import from services.py

@shared_task
def update_disposable_incomes():
    """
    Celery task to update the monthly disposable income for all users.
    """
    for user_profile in UserProfile.objects.all():
        calculate_and_update_disposable_income(user_profile)