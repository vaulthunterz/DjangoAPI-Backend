"""
AI Service Factory Module

This module provides a factory for creating and accessing AI services.
"""
import logging
from typing import Dict, Any, Optional, Type

from .service import AIService
from .expense_service import ExpenseAIService
from .investment_service import InvestmentAIService
from .gemini_service import GeminiAIService

# Configure logging
logger = logging.getLogger(__name__)

class AIServiceFactory:
    """
    Factory class for creating and accessing AI services.
    
    This class provides a centralized way to access all AI services in the application.
    It ensures that services are initialized only once and reused across the application.
    """
    
    _instance = None
    _services = {}
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AIServiceFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the factory if not already initialized."""
        if not self._initialized:
            self._services = {}
            self._initialized = True
    
    def get_service(self, service_type: str) -> AIService:
        """
        Get an AI service of the specified type.
        
        Args:
            service_type: Type of AI service ('expense', 'investment', or 'gemini')
            
        Returns:
            AIService: The requested AI service
            
        Raises:
            ValueError: If the service type is unknown
        """
        # Check if service already exists
        if service_type in self._services:
            return self._services[service_type]
        
        # Create and initialize service
        service = self._create_service(service_type)
        
        if service:
            # Initialize the service
            success = service.initialize()
            
            if not success:
                logger.warning(f"Failed to initialize {service_type} AI service")
            
            # Store service for reuse
            self._services[service_type] = service
            
            return service
        
        raise ValueError(f"Unknown AI service type: {service_type}")
    
    def _create_service(self, service_type: str) -> Optional[AIService]:
        """
        Create an AI service of the specified type.
        
        Args:
            service_type: Type of AI service
            
        Returns:
            Optional[AIService]: The created AI service, or None if the type is unknown
        """
        if service_type == 'expense':
            return ExpenseAIService()
        elif service_type == 'investment':
            return InvestmentAIService()
        elif service_type == 'gemini':
            return GeminiAIService()
        else:
            logger.error(f"Unknown AI service type: {service_type}")
            return None
    
    def get_all_services(self) -> Dict[str, AIService]:
        """
        Get all available AI services.
        
        Returns:
            Dict[str, AIService]: Dictionary of all available AI services
        """
        # Initialize all services if not already done
        for service_type in ['expense', 'investment', 'gemini']:
            if service_type not in self._services:
                try:
                    self.get_service(service_type)
                except Exception as e:
                    logger.error(f"Error initializing {service_type} service: {str(e)}")
        
        return self._services
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about all AI services.
        
        Returns:
            Dict[str, Any]: Information about all AI services
        """
        services = self.get_all_services()
        
        info = {}
        for service_type, service in services.items():
            info[service_type] = service.get_model_info()
        
        return info


# Create a singleton instance
ai_service_factory = AIServiceFactory()

# Convenience functions
def get_expense_ai_service() -> ExpenseAIService:
    """Get the expense AI service."""
    return ai_service_factory.get_service('expense')

def get_investment_ai_service() -> InvestmentAIService:
    """Get the investment AI service."""
    return ai_service_factory.get_service('investment')

def get_gemini_ai_service() -> GeminiAIService:
    """Get the Gemini AI service."""
    return ai_service_factory.get_service('gemini')
