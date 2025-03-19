# investment/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_nested import routers
from . import views

# --- Routers ---

router = DefaultRouter()
router.register(r'userprofiles', views.UserProfileViewSet)
router.register(r'portfolios', views.PortfolioViewSet)
router.register(r'financial-assets', views.FinancialAssetViewSet)  # Usually read-only
router.register(r'recommendations', views.RecommendationViewSet)
router.register(r'alerts', views.AlertViewSet)

# Nested router for portfolio items (within portfolios)
portfolio_router = routers.NestedDefaultRouter(router, r'portfolios', lookup='portfolio')
portfolio_router.register(r'items', views.PortfolioItemViewSet, basename='portfolio-items')

# --- URL Patterns ---

urlpatterns = [
    path('', include(router.urls)),
    path('', include(portfolio_router.urls)),  # Include nested routes
]