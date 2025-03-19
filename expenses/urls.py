from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    TransactionViewSet, CategoryViewSet, SubCategoryViewSet,
    CategoryLookupView, SubCategoryLookupView,
    ChatbotView, ModelMetricsView, RetrainModelView
)
from .gemini_prediction import GeminiPredictionView
from .custom_prediction import CustomModelPredictionView

router = DefaultRouter()
router.register(r'transactions', TransactionViewSet)
router.register(r'categories', CategoryViewSet)
router.register(r'subcategories', SubCategoryViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('category-lookup/', CategoryLookupView.as_view(), name='category-lookup'),
    path('subcategory-lookup/', SubCategoryLookupView.as_view(), name='subcategory-lookup'),
    path('predict/gemini/', GeminiPredictionView.as_view(), name='gemini-predict'),
    path('predict/custom/', CustomModelPredictionView.as_view(), name='custom-predict'),
    path('chatbot/', ChatbotView.as_view(), name='chatbot'),
    path('model-metrics/', ModelMetricsView.as_view(), name='model-metrics'),
    path('model/retrain/', RetrainModelView.as_view(), name='model-retrain'),
]