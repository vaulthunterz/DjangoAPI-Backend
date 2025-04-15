"""
AI Service Django App Configuration
"""
from django.apps import AppConfig


class AIServiceConfig(AppConfig):
    """
    Django app configuration for the AI service.
    """
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_service'
    verbose_name = 'AI Service'
    
    def ready(self):
        """
        Initialize the AI service when the app is ready.
        """
        # Import the factory to initialize it
        from .factory import ai_service_factory
        
        # Log that the AI service is ready
        import logging
        logger = logging.getLogger(__name__)
        logger.info("AI Service initialized")
