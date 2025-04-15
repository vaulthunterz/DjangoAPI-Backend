"""
URL patterns for the AI service API.
"""
from django.urls import path
from .views import (
    ExpensePredictionView,
    ExpenseCustomPredictionView,
    ExpenseModelMetricsView,
    ExpenseModelTrainingView,
    GeminiPredictionView,
    GeminiChatView,
    InvestmentPredictionView,
    InvestmentPortfolioAnalysisView,
    InvestmentExpenseRecommendationsView,
    AIServiceInfoView
)

urlpatterns = [
    # Expense AI endpoints
    path('expense/predict/', ExpensePredictionView.as_view(), name='expense-predict'),
    path('expense/predict/custom/', ExpenseCustomPredictionView.as_view(), name='expense-predict-custom'),
    path('expense/metrics/', ExpenseModelMetricsView.as_view(), name='expense-metrics'),
    path('expense/train/', ExpenseModelTrainingView.as_view(), name='expense-train'),
    
    # Gemini AI endpoints
    path('gemini/predict/', GeminiPredictionView.as_view(), name='gemini-predict'),
    path('gemini/chat/', GeminiChatView.as_view(), name='gemini-chat'),
    
    # Investment AI endpoints
    path('investment/predict/', InvestmentPredictionView.as_view(), name='investment-predict'),
    path('investment/analyze/<int:portfolio_id>/', InvestmentPortfolioAnalysisView.as_view(), name='investment-analyze'),
    path('investment/expense-recommendations/<int:user_id>/', InvestmentExpenseRecommendationsView.as_view(), name='investment-expense-recommendations'),
    
    # General AI service info
    path('info/', AIServiceInfoView.as_view(), name='ai-service-info'),
]
