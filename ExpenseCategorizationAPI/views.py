from django.http import JsonResponse
from django.utils import timezone # For timestamp

def api_root_status(request):
    """
    A simple view for the API root, returning a status message and timestamp.
    """
    status_data = {
        "status": "Financial Management API is running",
        "version": "1.0.0", # You can manage this version string as you like
        "timestamp": timezone.now().isoformat(),
        "documentation_urls": {
            "swagger_ui": "/swagger/",
            "redoc": "/redoc/"
        }
    }
    return JsonResponse(status_data)
    
