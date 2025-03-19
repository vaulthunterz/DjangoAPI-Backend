from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator


class UserProfile(models.Model):
    RISK_TOLERANCE_CHOICES = [
        (1, 'Low'),
        (2, 'Medium'),
        (3, 'High'),
    ]
    INVESTMENT_EXPERIENCE_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    INVESTMENT_TIMELINE_CHOICES = [
        ('short', 'Short-term (< 3 years)'),
        ('mid', 'Mid-term (3-10 years)'),
        ('long', 'Long-term (> 10 years)'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userprofile')
    risk_tolerance = models.IntegerField(choices=RISK_TOLERANCE_CHOICES)
    investment_experience = models.CharField(max_length=20, choices=INVESTMENT_EXPERIENCE_CHOICES)
    monthly_disposable_income = models.DecimalField(
        max_digits=15, decimal_places=2, blank=True, null=True,
        validators=[MinValueValidator(0)]
    )  # Calculated field
    investment_timeline = models.CharField(max_length=20, choices=INVESTMENT_TIMELINE_CHOICES)
    investment_goals = models.CharField(max_length=200)  # Could be a comma-separated list, or a separate model
    investment_preference = models.CharField(max_length=200, blank = True, null = True) #Could be comma separated list, or a seperate model.

    def __str__(self):
        return self.user.username


class Portfolio(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='portfolios')
    name = models.CharField(max_length=100)
    total_amount = models.DecimalField(max_digits=15, decimal_places=2, validators=[MinValueValidator(0)])
    risk_level = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)])  # Example: 1-10 scale
    creation_date = models.DateTimeField(auto_now_add=True)
    description = models.TextField(blank=True, null=True)
    strategy = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.name} ({self.user.user.username})"

    class Meta:
        ordering = ['-creation_date']


class FinancialAsset(models.Model):
    ASSET_TYPE_CHOICES = [
      ('Stock', 'Stock'),
      ('Bond', 'Bond'),
      ('ETF', 'ETF'),
      ('Crypto', 'Cryptocurrency'),
      ('Other', 'Other'),
    ]
    name = models.CharField(max_length=200)
    symbol = models.CharField(max_length=20, unique=True)  # e.g., "SAFCOM.NR"
    description = models.TextField(blank=True, null=True)
    sector = models.CharField(max_length=100, blank=True, null=True)
    asset_type = models.CharField(max_length=50, choices=ASSET_TYPE_CHOICES)
    risk_level = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)]) # Example scale

    def __str__(self):
        return self.symbol

    class Meta:
        ordering = ['name']


class PortfolioItem(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='items')
    asset_name = models.CharField(max_length=100) # Use asset's name and not a foreign key, for simplicity
    quantity = models.DecimalField(max_digits=15, decimal_places=4, validators=[MinValueValidator(0)])
    buy_price = models.DecimalField(max_digits=15, decimal_places=2, validators=[MinValueValidator(0)])
    purchase_time = models.DateTimeField()

    def __str__(self):
        return f"{self.asset_name} ({self.portfolio.name})"


class Recommendation(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='recommendations')
    message = models.CharField(max_length=200)
    timestamp = models.DateTimeField(auto_now_add=True)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.SET_NULL, null=True, blank=True, related_name='recommendations')
    financial_asset = models.ForeignKey(FinancialAsset, on_delete=models.SET_NULL, null=True, blank=True, related_name='recommendations') #Link recommendations to FinancialAsset

    def __str__(self):
        return f"Recommendation for {self.user.user.username} at {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']

class Alert(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='alerts')
    message = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    portfolio_item = models.ForeignKey(PortfolioItem, on_delete=models.SET_NULL, null=True, blank=True, related_name='alerts')  # Link to PortfolioItem

    def __str__(self):
        return f"Alert for {self.user.user.username} at {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']