from django.urls import path
from .views import TransactionListCreateView, CategoryPredictionView, CustomModelPredictionView, RetrainModelView, \
    TestAPIView

urlpatterns = [
    path('transactions/', TransactionListCreateView.as_view(), name='transaction-list-create'),
    path('predict/', CategoryPredictionView.as_view(), name='predict-category'),
    path('predict_custom/', CustomModelPredictionView.as_view(), name='predict-custom-category'),
    path('retrain/', RetrainModelView.as_view(), name='retrain-model'),

    path('test/', TestAPIView.as_view(), name='test-api'),
]