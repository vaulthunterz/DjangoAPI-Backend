from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.db.models import Sum, Count
from .models import (
    UserProfile, InvestmentQuestionnaire, Portfolio,
    PortfolioItem, MoneyMarketFund, Recommendation, Alert
)

class PortfolioItemInline(admin.TabularInline):
    model = PortfolioItem
    extra = 1
    fields = ('asset_name', 'quantity', 'buy_price', 'purchase_time')

class RecommendationInline(admin.TabularInline):
    model = Recommendation
    extra = 0
    fields = ('message', 'timestamp', 'recommendation_type', 'confidence_score')
    readonly_fields = ('timestamp',)
    max_num = 5

class AlertInline(admin.TabularInline):
    model = Alert
    extra = 0
    fields = ('message', 'timestamp')
    readonly_fields = ('timestamp',)
    max_num = 5

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user_link', 'risk_tolerance_display', 'investment_experience',
                   'investment_timeline', 'monthly_disposable_income')
    list_filter = ('risk_tolerance', 'investment_experience', 'investment_timeline')
    search_fields = ('user__username', 'user__email', 'investment_goals')
    fieldsets = (
        ('User Information', {
            'fields': ('user',)
        }),
        ('Investment Profile', {
            'fields': ('risk_tolerance', 'investment_experience', 'investment_timeline',
                      'investment_goals', 'investment_preference', 'monthly_disposable_income')
        }),
    )
    inlines = [RecommendationInline]

    def user_link(self, obj):
        url = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.username)
    user_link.short_description = 'User'

    def risk_tolerance_display(self, obj):
        colors = {1: 'green', 2: 'orange', 3: 'red'}
        labels = {1: 'Low', 2: 'Medium', 3: 'High'}
        return format_html('<span style="color: {};">{}</span>',
                          colors.get(obj.risk_tolerance, 'black'),
                          labels.get(obj.risk_tolerance, 'Unknown'))
    risk_tolerance_display.short_description = 'Risk Tolerance'

@admin.register(InvestmentQuestionnaire)
class InvestmentQuestionnaireAdmin(admin.ModelAdmin):
    list_display = ('user_link', 'primary_goal', 'investment_timeframe', 'risk_level', 'created_at')
    list_filter = ('primary_goal', 'investment_timeframe', 'market_drop_reaction', 'investment_preference')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at', 'updated_at')
    date_hierarchy = 'created_at'
    fieldsets = (
        ('User Information', {
            'fields': ('user', 'created_at', 'updated_at')
        }),
        ('Financial Situation', {
            'fields': ('annual_income_range', 'monthly_savings_range', 'emergency_fund_months', 'debt_situation')
        }),
        ('Investment Goals', {
            'fields': ('primary_goal', 'investment_timeframe', 'monthly_investment')
        }),
        ('Risk Assessment', {
            'fields': ('market_drop_reaction', 'investment_preference', 'loss_tolerance', 'risk_comfort_scenario')
        }),
        ('Investment Knowledge', {
            'fields': ('investment_knowledge', 'investment_experience_years', 'previous_investments')
        }),
        ('Preferences', {
            'fields': ('preferred_investment_types', 'ethical_preferences', 'sector_preferences')
        }),
        ('Additional Information', {
            'fields': ('financial_dependents', 'income_stability', 'major_expenses_planned')
        }),
    )

    def user_link(self, obj):
        url = reverse("admin:auth_user_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.username)
    user_link.short_description = 'User'

    def risk_level(self, obj):
        # Simple risk calculation based on investment_preference
        risk_mapping = {
            'very_safe': 'Very Low',
            'conservative': 'Low',
            'balanced': 'Medium',
            'growth': 'High',
            'aggressive': 'Very High',
            None: 'Unknown'
        }
        risk_colors = {
            'Very Low': 'green',
            'Low': 'lightgreen',
            'Medium': 'orange',
            'High': 'orangered',
            'Very High': 'red',
            'Unknown': 'gray'
        }
        risk = risk_mapping.get(obj.investment_preference, 'Unknown')
        return format_html('<span style="color: {};">{}</span>', risk_colors[risk], risk)
    risk_level.short_description = 'Risk Level'

@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ('name', 'user_link', 'total_amount', 'risk_level_display', 'creation_date', 'item_count')
    list_filter = ('risk_level', 'creation_date')
    search_fields = ('name', 'user__user__username', 'description')
    date_hierarchy = 'creation_date'
    inlines = [PortfolioItemInline, RecommendationInline]

    def user_link(self, obj):
        url = reverse("admin:investment_userprofile_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.user.username)
    user_link.short_description = 'User'

    def risk_level_display(self, obj):
        # Color-code risk level from 1-10
        if obj.risk_level <= 3:
            color = 'green'
        elif obj.risk_level <= 6:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{}/10</span>', color, obj.risk_level)
    risk_level_display.short_description = 'Risk Level'

    def item_count(self, obj):
        return obj.items.count()
    item_count.short_description = 'Items'

@admin.register(PortfolioItem)
class PortfolioItemAdmin(admin.ModelAdmin):
    list_display = ('asset_name', 'portfolio_link', 'user_link', 'quantity', 'buy_price', 'total_value', 'purchase_time')
    list_filter = ('portfolio', 'purchase_time')
    search_fields = ('asset_name', 'portfolio__name', 'portfolio__user__user__username')
    date_hierarchy = 'purchase_time'

    def portfolio_link(self, obj):
        url = reverse("admin:investment_portfolio_change", args=[obj.portfolio.id])
        return format_html('<a href="{}">{}</a>', url, obj.portfolio.name)
    portfolio_link.short_description = 'Portfolio'

    def user_link(self, obj):
        url = reverse("admin:investment_userprofile_change", args=[obj.portfolio.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.portfolio.user.user.username)
    user_link.short_description = 'User'

    def total_value(self, obj):
        total = obj.quantity * obj.buy_price
        # Format the total as a string first
        formatted_value = f"${float(total):.2f}"
        return format_html('{}', formatted_value)
    total_value.short_description = 'Total Value'

@admin.register(MoneyMarketFund)
class MoneyMarketFundAdmin(admin.ModelAdmin):
    list_display = ('name', 'symbol', 'fund_manager', 'risk_level_display', 'min_investment', 'expected_returns')
    list_filter = ('risk_level', 'fund_manager')
    search_fields = ('name', 'symbol', 'fund_manager', 'description')
    fieldsets = (
        ('Fund Information', {
            'fields': ('name', 'symbol', 'description', 'fund_manager', 'risk_level')
        }),
        ('Financial Details', {
            'fields': ('min_investment', 'expected_returns', 'liquidity', 'fees')
        }),
        ('Additional Information', {
            'fields': ('inception_date', 'fund_size', 'website', 'contact_info')
        }),
    )

    def risk_level_display(self, obj):
        # Color-code risk level from 1-10
        if obj.risk_level <= 3:
            color = 'green'
        elif obj.risk_level <= 6:
            color = 'orange'
        else:
            color = 'red'
        return format_html('<span style="color: {};">{}/10</span>', color, obj.risk_level)
    risk_level_display.short_description = 'Risk Level'

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user_link', 'message_preview', 'recommendation_type', 'confidence_score_display', 'timestamp')
    list_filter = ('recommendation_type', 'timestamp')
    search_fields = ('user__user__username', 'message', 'explanation')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'

    def user_link(self, obj):
        url = reverse("admin:investment_userprofile_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.user.username)
    user_link.short_description = 'User'

    def message_preview(self, obj):
        if len(obj.message) > 50:
            return obj.message[:50] + '...'
        return obj.message
    message_preview.short_description = 'Message'

    def confidence_score_display(self, obj):
        # Color-code confidence score
        if obj.confidence_score < 0.4:
            color = 'red'
        elif obj.confidence_score < 0.7:
            color = 'orange'
        else:
            color = 'green'
        # Format the confidence score as a string first
        percentage = float(obj.confidence_score) * 100
        formatted_value = f"{percentage:.1f}%"
        return format_html('<span style="color: {};">{}</span>', color, formatted_value)
    confidence_score_display.short_description = 'Confidence'

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('user_link', 'message_preview', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('user__user__username', 'message')
    readonly_fields = ('timestamp',)
    date_hierarchy = 'timestamp'

    def user_link(self, obj):
        url = reverse("admin:investment_userprofile_change", args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.user.username)
    user_link.short_description = 'User'

    def message_preview(self, obj):
        if len(obj.message) > 50:
            return obj.message[:50] + '...'
        return obj.message
    message_preview.short_description = 'Message'
