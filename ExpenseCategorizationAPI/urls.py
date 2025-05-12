# /srv/financial-app/ExpenseCategorizationAPI/urls.py
from django.contrib import admin
from django.urls import path, include, re_path
from core.admin import custom_admin_site
# from django.views.generic import RedirectView # No longer needed for root redirect
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Import views
from .views import api_root_status, home_view

# Import model metrics views directly
from expenses.model_metrics_views import (
    model_metrics_view,
    model_metrics_api,
    train_model_view,
    training_status_api
)

# Swagger/OpenAPI configuration
schema_view = get_schema_view(
    openapi.Info(
        title="Expense Categorization API",
        default_version='v1',
        description="API for expense categorization and investment recommendations",
        terms_of_service="https://www.example.com/terms/",
        contact=openapi.Contact(email="contact@example.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('manage-site-fintrackke/', admin.site.urls), # Your obscured admin URL
    path('admin-dashboard/', custom_admin_site.urls), # Custom admin dashboard

    # Import app URLs
    path('api/expenses/', include('expenses.urls')),
    path('api/investment/', include('investment.urls')),
    path('api/ai/', include('ai_service.urls')),
    path('api/auth/', include([
        path('user/', include('expenses.urls')),
    ])),

    # Web interface URLs
    path('', home_view, name='home'),
    path('api/', api_root_status, name='api-root-status'),

    # Direct access to model metrics page
    path('expenses/model-metrics/', model_metrics_view, name='model-metrics'),
    path('expenses/model_metrics/', model_metrics_view, name='model_metrics'),

    # Direct access to model metrics API
    path('expenses/model-metrics-api/', model_metrics_api, name='model-metrics-api'),
    path('expenses/model_metrics_api/', model_metrics_api, name='model_metrics_api'),

    # Direct access to model training endpoints
    path('expenses/model-train/', train_model_view, name='model-train'),
    path('expenses/model_train/', train_model_view, name='model_train'),
    path('expenses/training-status/', training_status_api, name='training-status'),

    # Swagger/OpenAPI documentation URLs
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

