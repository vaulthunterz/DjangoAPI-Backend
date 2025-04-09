# --- Utility Functions ---
import os
from datetime import timedelta

import requests
from django.db import models
from django.utils import timezone
from dotenv import load_dotenv

from expenses.models import Transaction


load_dotenv()
FINANCIAL_DATA_API_KEY = os.getenv("FINANCIAL_DATA_API_KEY")
FINANCIAL_DATA_API_URL = os.getenv("FINANCIAL_DATA_API_URL")

def fetch_market_data(symbol):
    """Fetches market data for a given symbol (e.g., stock ticker)."""
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',  # Example function (adjust based on your API)
        'symbol': symbol,
        'apikey': FINANCIAL_DATA_API_KEY,
    }
    try:
        response = requests.get(FINANCIAL_DATA_API_URL, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching market data: {e}")  # Log the error
        return None

def calculate_and_update_disposable_income(user_profile):
    """Calculates and updates the monthly disposable income for a user."""
    today = timezone.now()
    one_month_ago = today - timedelta(days=30)

    # 1. Estimate Income (Example: Using M-PESA deposits as a proxy)
    total_income = Transaction.objects.filter(
        user=user_profile.user,
        time_of_transaction__gte=one_month_ago,
        category__name__in=["Income", "Salary", "Deposits"]  # Adjust category names!
    ).aggregate(models.Sum('amount'))['amount__sum'] or 0

    # 2. Get Total Expenses
    total_expenses = Transaction.objects.filter(
        user=user_profile.user,
        time_of_transaction__gte=one_month_ago,
        category__name__in=["Food", "Rent", "Utilities", "Transport", "Entertainment"]  # Your expense categories
    ).aggregate(models.Sum('amount'))['amount__sum'] or 0

    # 3. Calculate and Update
    disposable_income = total_income - total_expenses
    user_profile.monthly_disposable_income = disposable_income
    user_profile.save()