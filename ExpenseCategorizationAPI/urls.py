from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/expenses/', include('expenses.urls')),
    # Comment out investment URLs for now - will add back later when fully fixed
    # path('api/investment/', include('investment.urls')),
    # path('', RedirectView.as_view(url='/api/transactions/'), name='api-root'),
]