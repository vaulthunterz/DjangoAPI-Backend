from django.contrib import admin
from django.utils.html import format_html
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Sum
from django.contrib.auth.models import User
from expenses.models import Transaction, Category
from investment.models import Portfolio, UserProfile, MoneyMarketFund, Recommendation
import json
from datetime import datetime, timedelta

class CustomAdminSite(admin.AdminSite):
    site_header = 'FinTrack Administration'
    site_title = 'FinTrack Admin Portal'
    index_title = 'FinTrack Administration'
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('dashboard/', self.admin_view(self.dashboard_view), name='dashboard'),
        ]
        return custom_urls + urls
    
    def index(self, request, extra_context=None):
        """Override the default index to add dashboard stats"""
        # Get basic stats
        stats = self.get_app_stats()
        
        # Add stats to the context
        context = extra_context or {}
        context.update({
            'stats': stats,
            'has_dashboard_permission': True,
        })
        return super().index(request, context)
    
    def dashboard_view(self, request):
        """Custom dashboard view with detailed statistics"""
        context = {
            **self.each_context(request),
            'title': 'Dashboard',
            'stats': self.get_app_stats(),
            'expense_data': self.get_expense_data(),
            'investment_data': self.get_investment_data(),
            'user_activity': self.get_user_activity(),
        }
        return render(request, 'admin/dashboard.html', context)
    
    def get_app_stats(self):
        """Get basic application statistics"""
        # User stats
        total_users = User.objects.count()
        active_users = User.objects.filter(is_active=True).count()
        
        # Transaction stats
        total_transactions = Transaction.objects.count()
        total_expense = Transaction.objects.filter(is_expense=True).aggregate(Sum('amount'))['amount__sum'] or 0
        total_income = Transaction.objects.filter(is_expense=False).aggregate(Sum('amount'))['amount__sum'] or 0
        
        # Investment stats
        total_portfolios = Portfolio.objects.count()
        total_invested = Portfolio.objects.aggregate(Sum('total_amount'))['total_amount__sum'] or 0
        total_funds = MoneyMarketFund.objects.count()
        
        # Recent activity
        recent_transactions = Transaction.objects.order_by('-time_of_transaction')[:5]
        recent_recommendations = Recommendation.objects.order_by('-timestamp')[:5]
        
        return {
            'users': {
                'total': total_users,
                'active': active_users,
                'inactive': total_users - active_users,
            },
            'transactions': {
                'total': total_transactions,
                'expense': total_expense,
                'income': total_income,
                'balance': total_income - total_expense,
                'recent': recent_transactions,
            },
            'investments': {
                'portfolios': total_portfolios,
                'total_invested': total_invested,
                'funds': total_funds,
                'recent_recommendations': recent_recommendations,
            }
        }
    
    def get_expense_data(self):
        """Get detailed expense data for charts"""
        # Category distribution
        categories = Category.objects.annotate(
            transaction_count=Count('transactions'),
            total_amount=Sum('transactions__amount')
        ).values('name', 'transaction_count', 'total_amount')
        
        # Time series data (last 30 days)
        today = datetime.now().date()
        thirty_days_ago = today - timedelta(days=30)
        
        daily_expenses = Transaction.objects.filter(
            is_expense=True,
            time_of_transaction__date__gte=thirty_days_ago
        ).values('time_of_transaction__date').annotate(
            total=Sum('amount')
        ).order_by('time_of_transaction__date')
        
        # Format for charts
        category_labels = [c['name'] for c in categories]
        category_amounts = [float(c['total_amount'] or 0) for c in categories]
        category_counts = [c['transaction_count'] for c in categories]
        
        date_labels = [(thirty_days_ago + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(31)]
        daily_amounts = [0] * 31
        
        for expense in daily_expenses:
            day_index = (expense['time_of_transaction__date'] - thirty_days_ago).days
            if 0 <= day_index < 31:
                daily_amounts[day_index] = float(expense['total'])
        
        return {
            'categories': {
                'labels': category_labels,
                'amounts': category_amounts,
                'counts': category_counts,
            },
            'time_series': {
                'labels': date_labels,
                'amounts': daily_amounts,
            }
        }
    
    def get_investment_data(self):
        """Get detailed investment data for charts"""
        # Risk distribution
        risk_distribution = UserProfile.objects.values('risk_tolerance').annotate(
            count=Count('id')
        ).order_by('risk_tolerance')
        
        # Portfolio performance (simplified)
        portfolios = Portfolio.objects.values('name', 'total_amount', 'risk_level')
        
        # Format for charts
        risk_labels = ['Low', 'Medium', 'High']
        risk_counts = [0, 0, 0]
        
        for risk in risk_distribution:
            if risk['risk_tolerance'] and 1 <= risk['risk_tolerance'] <= 3:
                risk_counts[risk['risk_tolerance']-1] = risk['count']
        
        portfolio_labels = [p['name'] for p in portfolios]
        portfolio_amounts = [float(p['total_amount']) for p in portfolios]
        portfolio_risks = [p['risk_level'] for p in portfolios]
        
        return {
            'risk_distribution': {
                'labels': risk_labels,
                'counts': risk_counts,
            },
            'portfolios': {
                'labels': portfolio_labels,
                'amounts': portfolio_amounts,
                'risks': portfolio_risks,
            }
        }
    
    def get_user_activity(self):
        """Get user activity data"""
        # Active users by month
        today = datetime.now().date()
        six_months_ago = today - timedelta(days=180)
        
        # Get transaction counts by user
        user_transaction_counts = Transaction.objects.values('user__username').annotate(
            count=Count('id')
        ).order_by('-count')[:10]
        
        # Get portfolio counts by user
        user_portfolio_counts = Portfolio.objects.values('user__user__username').annotate(
            count=Count('id')
        ).order_by('-count')[:10]
        
        return {
            'top_transaction_users': {
                'labels': [u['user__username'] for u in user_transaction_counts],
                'counts': [u['count'] for u in user_transaction_counts],
            },
            'top_portfolio_users': {
                'labels': [u['user__user__username'] for u in user_portfolio_counts],
                'counts': [u['count'] for u in user_portfolio_counts],
            }
        }

# Create custom admin site
custom_admin_site = CustomAdminSite(name='custom_admin')

# Register models with the custom admin site
# You would need to register all your models here
# For example:
# from django.contrib.auth.models import User, Group
# from django.contrib.auth.admin import UserAdmin, GroupAdmin
# custom_admin_site.register(User, UserAdmin)
# custom_admin_site.register(Group, GroupAdmin)
