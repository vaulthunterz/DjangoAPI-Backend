from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TransactionViewSet, CategoryViewSet, SubCategoryViewSet,
    CategoryLookupView, SubCategoryLookupView,
    ChangePasswordView
)

router = DefaultRouter()
router.register(r'transactions', TransactionViewSet)
router.register(r'categories', CategoryViewSet)
router.register(r'subcategories', SubCategoryViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('category-lookup/', CategoryLookupView.as_view(), name='category-lookup'),
    path('subcategory-lookup/', SubCategoryLookupView.as_view(), name='subcategory-lookup'),
    path('change-password/', ChangePasswordView.as_view(), name='change-password'),

    # The following endpoints have been moved to the AI service:
    # - predict/gemini/
    # - predict/custom/
    # - chatbot/
    # - model-metrics/
    # - model/retrain/
]