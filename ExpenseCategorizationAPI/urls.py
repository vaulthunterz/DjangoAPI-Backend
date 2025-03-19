from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/expenses/', include('expenses.urls')),
    # path('api/investment/', include('investment.urls')),
    # path('', RedirectView.as_view(url='/api/transactions/'), name='api-root'),
]