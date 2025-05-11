"""
URL configuration for the expenses app.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

# Import view classes directly from the views.py file
# Use absolute imports to avoid confusion with the views package
from expenses.views import (
    TransactionViewSet,
    CategoryViewSet,
    SubCategoryViewSet,
    CategoryLookupView,
    SubCategoryLookupView,
    ChangePasswordView
)

# Import model metrics views from the model_metrics_views.py file
from .model_metrics_views import (
    model_metrics_view,
    model_metrics_api,
    train_model_view,
    training_status_api
)

# Create a router for the viewsets
router = DefaultRouter()
router.register(r'transactions', TransactionViewSet)
router.register(r'categories', CategoryViewSet)
router.register(r'subcategories', SubCategoryViewSet)

# Define the URL patterns
urlpatterns = [
    path('', include(router.urls)),
    path('category-lookup/', CategoryLookupView.as_view(), name='category-lookup'),
    path('subcategory-lookup/', SubCategoryLookupView.as_view(), name='subcategory-lookup'),
    path('change-password/', ChangePasswordView.as_view(), name='change-password'),

    # Model metrics visualization
    path('model-metrics/', model_metrics_view, name='model-metrics'),
    path('model_metrics/', model_metrics_view, name='model_metrics'),  # Alternative URL with underscore
    path('api/model-metrics/', model_metrics_api, name='model-metrics-api'),

    # Model training endpoints
    path('model-train/', train_model_view, name='model-train'),
    path('model_train/', train_model_view, name='model_train'),  # Alternative URL with underscore
    path('api/training-status/', training_status_api, name='training-status-api'),

    # The following endpoints have been moved to the AI service:
    # - predict/gemini/
    # - predict/custom/
    # - chatbot/
    # - model/retrain/
]