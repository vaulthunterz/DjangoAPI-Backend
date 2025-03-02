from django.urls import path

from . import views
from .views import TransactionListCreateView, CategoryPredictionView, CustomModelPredictionView, RetrainModelView, \
    ChatbotView, TransactionUpdateView

urlpatterns = [
    path('transactions/', TransactionListCreateView.as_view(), name='transaction-list-create'),

    path('transactions/<int:pk>/', TransactionUpdateView.as_view(), name='transaction-update'),

    path('categories/', views.CategoryLookupView.as_view()),
    path('subcategories/', views.SubCategoryLookupView.as_view()),
    path('predict/', CategoryPredictionView.as_view(), name='predict-category'),
    path('predict_custom/', CustomModelPredictionView.as_view(), name='predict-custom-category'),
    path('retrain/', RetrainModelView.as_view(), name='retrain-model'),
    path('chatbot/', ChatbotView.as_view(), name="chatbot-view"),
]