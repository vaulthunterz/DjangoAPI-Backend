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


class InvestmentQuestionnaire(models.Model):
    """Model to store user responses to investment questionnaire"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='investment_questionnaire')
    
    # Financial Situation
    annual_income_range = models.CharField(max_length=50, choices=[
        ('0-30000', 'Less than $30,000'),
        ('30000-50000', '$30,000 - $50,000'),
        ('50000-80000', '$50,000 - $80,000'),
        ('80000-120000', '$80,000 - $120,000'),
        ('120000+', 'More than $120,000')
    ])
    monthly_savings_range = models.CharField(max_length=50, choices=[
        ('0-200', 'Less than $200'),
        ('200-500', '$200 - $500'),
        ('500-1000', '$500 - $1,000'),
        ('1000-2000', '$1,000 - $2,000'),
        ('2000+', 'More than $2,000')
    ])
    emergency_fund_months = models.CharField(max_length=50, choices=[
        ('0', 'No emergency fund'),
        ('1-3', '1-3 months of expenses'),
        ('3-6', '3-6 months of expenses'),
        ('6-12', '6-12 months of expenses'),
        ('12+', 'More than 12 months of expenses')
    ])
    debt_situation = models.CharField(max_length=50, choices=[
        ('none', 'No debt'),
        ('low', 'Low debt (manageable with current income)'),
        ('moderate', 'Moderate debt (working on paying off)'),
        ('high', 'High debt (struggling to manage)'),
        ('very_high', 'Very high debt (need debt management help)')
    ])
    
    # Investment Goals
    primary_goal = models.CharField(max_length=100, choices=[
        ('retirement', 'Planning for retirement'),
        ('education', 'Saving for education'),
        ('home', 'Buying a home'),
        ('wealth_building', 'Growing my wealth'),
        ('passive_income', 'Creating passive income'),
        ('emergency_fund', 'Building an emergency fund'),
        ('major_purchase', 'Saving for a major purchase'),
        ('other', 'Other goal')
    ])
    investment_timeframe = models.CharField(max_length=50, choices=[
        ('very_short', 'Less than 1 year'),
        ('short', '1-3 years'),
        ('medium', '3-5 years'),
        ('long', '5-10 years'),
        ('very_long', 'More than 10 years')
    ])
    monthly_investment = models.CharField(max_length=50, choices=[
        ('0-100', 'Less than $100'),
        ('100-300', '$100 - $300'),
        ('300-500', '$300 - $500'),
        ('500-1000', '$500 - $1,000'),
        ('1000+', 'More than $1,000')
    ])
    
    # Risk Assessment
    market_drop_reaction = models.CharField(max_length=50, choices=[
        ('sell_all', 'Sell all investments immediately'),
        ('sell_some', 'Sell some investments'),
        ('do_nothing', 'Do nothing and wait it out'),
        ('buy_more', 'Buy more while prices are low'),
        ('seek_advice', 'Seek professional advice')
    ])
    investment_preference = models.CharField(max_length=50, choices=[
        ('very_safe', 'Guaranteed returns with no risk'),
        ('conservative', 'Mostly safe investments with some growth potential'),
        ('balanced', 'Mix of safe and growth investments'),
        ('growth', 'Mostly growth with some risk protection'),
        ('aggressive', 'Maximum growth potential with higher risk')
    ])
    loss_tolerance = models.CharField(max_length=50, choices=[
        ('0-5', 'Less than 5%'),
        ('5-10', '5% - 10%'),
        ('10-20', '10% - 20%'),
        ('20-30', '20% - 30%'),
        ('30+', 'More than 30%')
    ])
    risk_comfort_scenario = models.CharField(max_length=50, choices=[
        ('scenario_1', 'Potential gain: 5%, Potential loss: 2%'),
        ('scenario_2', 'Potential gain: 10%, Potential loss: 5%'),
        ('scenario_3', 'Potential gain: 20%, Potential loss: 12%'),
        ('scenario_4', 'Potential gain: 30%, Potential loss: 20%'),
        ('scenario_5', 'Potential gain: 40%, Potential loss: 30%')
    ])
    
    # Investment Knowledge & Experience
    investment_knowledge = models.CharField(max_length=50, choices=[
        ('none', 'No knowledge or experience'),
        ('basic', 'Basic understanding of savings accounts and fixed deposits'),
        ('moderate', 'Familiar with stocks and mutual funds'),
        ('good', 'Good understanding of various investment products'),
        ('expert', 'Expert knowledge of financial markets')
    ])
    investment_experience_years = models.CharField(max_length=50, choices=[
        ('none', 'No experience'),
        ('0-2', 'Less than 2 years'),
        ('2-5', '2-5 years'),
        ('5-10', '5-10 years'),
        ('10+', 'More than 10 years')
    ])
    previous_investments = models.JSONField(default=list, help_text='List of investment types previously used')
    
    # Investment Preferences
    preferred_investment_types = models.JSONField(default=list, help_text='List of preferred investment types')
    ethical_preferences = models.JSONField(default=list, help_text='List of ethical investment preferences')
    sector_preferences = models.JSONField(default=list, help_text='List of preferred sectors')
    
    # Additional Information
    financial_dependents = models.CharField(max_length=50, choices=[
        ('none', 'No dependents'),
        ('1-2', '1-2 dependents'),
        ('3-4', '3-4 dependents'),
        ('5+', '5 or more dependents')
    ])
    income_stability = models.CharField(max_length=50, choices=[
        ('very_stable', 'Very stable income'),
        ('stable', 'Stable income'),
        ('somewhat_stable', 'Somewhat stable income'),
        ('unstable', 'Unstable income'),
        ('very_unstable', 'Very unstable income')
    ])
    major_expenses_planned = models.JSONField(default=list, help_text='List of planned major expenses in next 5 years')
    
    # System fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Questionnaire for {self.user.username}"


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


class MoneyMarketFund(models.Model):
    """Model representing Kenyan Money Market Funds"""
    name = models.CharField(max_length=200)
    symbol = models.CharField(max_length=20, unique=True)  # e.g., "CIC-MMF"
    description = models.TextField(blank=True, null=True)
    fund_manager = models.CharField(max_length=100)
    risk_level = models.IntegerField(validators=[MinValueValidator(1), MaxValueValidator(10)]) # 1-10 scale
    
    # Fund details
    min_investment = models.DecimalField(max_digits=15, decimal_places=2)
    expected_returns = models.CharField(max_length=100)
    liquidity = models.CharField(max_length=50, default='High')
    fees = models.CharField(max_length=100, default='0.5-1.5% annually')
    
    # Additional information
    inception_date = models.DateField(null=True, blank=True)
    fund_size = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    website = models.URLField(max_length=200, null=True, blank=True)
    contact_info = models.CharField(max_length=200, null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']
        verbose_name = 'Money Market Fund'
        verbose_name_plural = 'Money Market Funds'


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
    financial_asset = models.ForeignKey(MoneyMarketFund, on_delete=models.SET_NULL, null=True, blank=True, related_name='recommendations') #Link recommendations to FinancialAsset
    confidence_score = models.FloatField(default=0.0, validators=[MinValueValidator(0), MaxValueValidator(1)])
    recommendation_type = models.CharField(max_length=50, choices=[
        ('rule_based', 'Rule-Based'),
        ('ml_based', 'Machine Learning'),
        ('hybrid', 'Hybrid'),
        ('expense_based', 'Expense-Based')
    ], default='hybrid')
    explanation = models.TextField(blank=True, null=True)

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