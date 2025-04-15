"""
Gemini AI Service Module

This module provides AI functionality using Google's Gemini AI models.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Union
import google.generativeai as genai
from django.conf import settings

from .service import AIService

# Configure logging
logger = logging.getLogger(__name__)

class GeminiAIService(AIService):
    """
    AI service for Google's Gemini AI models.
    
    This service provides functionality for:
    - Text generation and completion
    - Category prediction
    - Chatbot responses
    """
    
    def __init__(self):
        """Initialize the Gemini AI service."""
        super().__init__()
        self.model = None
        self.api_key = None
        self.model_name = 'gemini-2.0-flash'  # Default model
    
    def initialize(self) -> bool:
        """
        Initialize the Gemini AI service and configure the API.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Get API key from settings
            self.api_key = getattr(settings, 'GOOGLE_AI_STUDIO_KEY', None)
            
            if not self.api_key:
                logger.warning("Gemini API key not found in settings")
                return False
            
            # Configure Gemini API
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            self.models = {
                self.model_name: True
            }
            
            self.initialized = True
            logger.info("Gemini AI service initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing Gemini AI service: {str(e)}")
            self.initialized = False
            return False
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a prediction or response using Gemini AI.
        
        Args:
            data: Dictionary containing input data
                 Required keys: 'prompt' or 'description'
            
        Returns:
            Dict[str, Any]: Generated response
        """
        self._ensure_initialized()
        
        # Get prompt from data
        prompt = data.get('prompt', data.get('description', ''))
        
        if not prompt:
            raise ValueError("Prompt or description is required")
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                'text': response.text,
                'model': self.model_name
            }
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            raise
    
    def predict_category(self, description: str, categories: List[str]) -> Dict[str, Any]:
        """
        Predict the category for a transaction description.
        
        Args:
            description: Transaction description
            categories: List of available categories
            
        Returns:
            Dict[str, Any]: Predicted category with confidence
        """
        self._ensure_initialized()
        
        if not description:
            raise ValueError("Description is required")
        
        if not categories:
            raise ValueError("Categories list is required")
        
        try:
            # Create prompt for Gemini
            prompt = f"""
            Classify the following expense description into exactly one of these categories:
            {', '.join(categories)}

            Description: {description}

            Respond ONLY with the matching category name in the format "Category - Subcategory".
            If unsure, respond with "Unknown - Other".
            """
            
            # Generate prediction
            response = self.model.generate_content(prompt)
            predicted_category = response.text.strip()
            
            # Parse the predicted category and subcategory
            if ' - ' not in predicted_category:
                predicted_category = "Unknown - Other"  # Fallback for invalid format
            
            category, subcategory = predicted_category.split(' - ', 1)
            
            return {
                'category': category,
                'subcategory': subcategory,
                'confidence': 0.85  # Gemini doesn't provide confidence scores, so we use a default
            }
        except Exception as e:
            logger.error(f"Error predicting category with Gemini: {str(e)}")
            raise
    
    def get_chatbot_response(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate a chatbot response using Gemini AI.
        
        Args:
            message: User message
            context: Optional conversation history
            
        Returns:
            Dict[str, Any]: Chatbot response
        """
        self._ensure_initialized()
        
        if not message:
            raise ValueError("Message is required")
        
        try:
            # Create prompt with context if provided
            if context:
                conversation = "\n".join([f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}" 
                                         for item in context])
                prompt = f"{conversation}\nUser: {message}\nAssistant:"
            else:
                prompt = f"User: {message}\nAssistant:"
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return {
                'text': response.text,
                'model': self.model_name
            }
        except Exception as e:
            logger.error(f"Error generating chatbot response: {str(e)}")
            raise
    
    def train(self, data: Any) -> Dict[str, Any]:
        """
        Train method is not applicable for Gemini AI service.
        
        Gemini models are pre-trained and cannot be trained directly.
        
        Args:
            data: Not used
            
        Returns:
            Dict[str, Any]: Status message
        """
        logger.warning("Gemini AI models cannot be trained directly")
        return {
            'status': 'not_applicable',
            'message': 'Gemini AI models are pre-trained and cannot be trained directly'
        }
    
    def set_model(self, model_name: str) -> bool:
        """
        Set the Gemini model to use.
        
        Args:
            model_name: Name of the Gemini model
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._ensure_initialized()
        
        try:
            # Check if model name is valid
            valid_models = ['gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']
            
            if model_name not in valid_models:
                logger.warning(f"Invalid model name: {model_name}. Using default model.")
                return False
            
            # Update model
            self.model_name = model_name
            self.model = genai.GenerativeModel(model_name)
            
            # Update models dictionary
            self.models = {model_name: True}
            
            logger.info(f"Gemini model set to {model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error setting Gemini model: {str(e)}")
            return False
