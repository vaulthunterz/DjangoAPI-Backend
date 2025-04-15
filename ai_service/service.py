"""
Base AI Service Module

This module provides a centralized interface for all AI-related functionality in the application.
It serves as the foundation for specialized AI services for different domains.
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class AIService(ABC):
    """
    Abstract base class for AI services.
    
    This class defines the common interface and functionality for all AI services
    in the application. Specialized services should inherit from this class.
    """
    
    def __init__(self):
        """Initialize the AI service."""
        self.models = {}
        self.initialized = False
        logger.info(f"Initializing {self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the AI service and load required models.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Dict[str, Any]:
        """
        Make a prediction using the AI service.
        
        Args:
            data: The input data for prediction.
            
        Returns:
            Dict[str, Any]: The prediction results.
        """
        pass
    
    @abstractmethod
    def train(self, data: Any) -> Dict[str, Any]:
        """
        Train the AI model with the provided data.
        
        Args:
            data: The training data.
            
        Returns:
            Dict[str, Any]: The training results.
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, Any]: Information about the loaded models.
        """
        return {
            "service_name": self.__class__.__name__,
            "initialized": self.initialized,
            "models": list(self.models.keys())
        }
    
    def _ensure_initialized(self) -> None:
        """
        Ensure that the service is initialized before use.
        
        Raises:
            RuntimeError: If the service is not initialized.
        """
        if not self.initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized. Call initialize() first.")
