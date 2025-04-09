# investment/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_nested import routers
from .views import (
    UserProfileViewSet, PortfolioViewSet, PortfolioItemViewSet,
    MoneyMarketFundViewSet, RecommendationViewSet, AlertViewSet,
    InvestmentQuestionnaireViewSet, ExpenseBasedRecommendationView
)

# --- Routers ---

router = DefaultRouter()
router.register(r'profiles', UserProfileViewSet, basename='profile')
router.register(r'questionnaires', InvestmentQuestionnaireViewSet, basename='questionnaire')
router.register(r'portfolios', PortfolioViewSet, basename='portfolio')
router.register(r'portfolio-items', PortfolioItemViewSet, basename='portfolio-item')
router.register(r'money-market-funds', MoneyMarketFundViewSet, basename='money-market-fund')
router.register(r'recommendations', RecommendationViewSet, basename='recommendation')
router.register(r'alerts', AlertViewSet, basename='alert')

# Nested router for portfolio items (within portfolios)
portfolio_router = routers.NestedDefaultRouter(router, r'portfolios', lookup='portfolio')
portfolio_router.register(r'items', PortfolioItemViewSet, basename='portfolio-items')

# --- URL Patterns ---

urlpatterns = [
    path('', include(router.urls)),
    path('', include(portfolio_router.urls)),  # Include nested routes
    path('expense-based-recommendations/', ExpenseBasedRecommendationView.as_view(), name='expense-based-recommendations'),
]