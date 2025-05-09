# /srv/financial-app/ExpenseCategorizationAPI/urls.py
from django.contrib import admin
from django.urls import path, include, re_path
# from django.views.generic import RedirectView # No longer needed for root redirect
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

# Import the new root status view
from .views import api_root_status

# Swagger/OpenAPI configuration (remains the same)
schema_view = get_schema_view(
    openapi.Info(
        title="Expense Categorization API",
        default_version='v1',
        description="API for expense categorization and investment recommendations",
        terms_of_service="https://www.example.com/terms/", # Replace with actual
        contact=openapi.Contact(email="contact@example.com"), # Replace with actual
        license=openapi.License(name="BSD License"), # Or your chosen license
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('manage-site-fintrackke/', admin.site.urls), # Your obscured admin URL
    path('api/expenses/', include('expenses.urls')),
    path('api/investment/', include('investment.urls')),
    path('api/ai/', include('ai_service.urls')),

    # Root URL now points to the API status view
    path('', api_root_status, name='api-root-status'),

    # Swagger/OpenAPI documentation URLs (still accessible at these specific paths)
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

