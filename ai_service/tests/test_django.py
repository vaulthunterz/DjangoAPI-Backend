"""
Django test suite for AI service tests.
"""
from django.test import TestCase
from unittest.mock import patch, MagicMock

from ai_service.factory import AIServiceFactory
from ai_service.service import AIService
from ai_service.expense_service import ExpenseAIService
from ai_service.investment_service import InvestmentAIService
from ai_service.gemini_service import GeminiAIService


class TestAIServiceFactory(TestCase):
    """Test cases for the AIServiceFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance and services
        AIServiceFactory._instance = None
        AIServiceFactory._services = {}
        
        # Create a new factory instance for each test
        self.factory = AIServiceFactory()
    
    def test_singleton_pattern(self):
        """Test that the factory follows the singleton pattern."""
        # Create two factory instances
        factory1 = AIServiceFactory()
        factory2 = AIServiceFactory()
        
        # They should be the same instance
        self.assertIs(factory1, factory2)
    
    @patch('ai_service.expense_service.ExpenseAIService.initialize')
    def test_get_expense_service(self, mock_initialize):
        """Test getting the expense service."""
        # Mock the initialize method to return True
        mock_initialize.return_value = True
        
        # Get the expense service
        service = self.factory.get_service('expense')
        
        # Check that we got the right type of service
        self.assertIsInstance(service, ExpenseAIService)
        
        # Check that initialize was called
        mock_initialize.assert_called_once()
        
        # Get the service again
        service2 = self.factory.get_service('expense')
        
        # Should be the same instance
        self.assertIs(service, service2)
        
        # Initialize should only be called once
        mock_initialize.assert_called_once()
    
    @patch('ai_service.investment_service.InvestmentAIService.initialize')
    def test_get_investment_service(self, mock_initialize):
        """Test getting the investment service."""
        # Mock the initialize method to return True
        mock_initialize.return_value = True
        
        # Get the investment service
        service = self.factory.get_service('investment')
        
        # Check that we got the right type of service
        self.assertIsInstance(service, InvestmentAIService)
        
        # Check that initialize was called
        mock_initialize.assert_called_once()
    
    @patch('ai_service.gemini_service.GeminiAIService.initialize')
    def test_get_gemini_service(self, mock_initialize):
        """Test getting the Gemini service."""
        # Mock the initialize method to return True
        mock_initialize.return_value = True
        
        # Get the Gemini service
        service = self.factory.get_service('gemini')
        
        # Check that we got the right type of service
        self.assertIsInstance(service, GeminiAIService)
        
        # Check that initialize was called
        mock_initialize.assert_called_once()
    
    def test_get_unknown_service(self):
        """Test getting an unknown service type."""
        # Try to get an unknown service
        with self.assertRaises(ValueError):
            self.factory.get_service('unknown')


class MockAIService(AIService):
    """Mock implementation of AIService for testing."""
    
    def initialize(self):
        self.initialized = True
        self.models = {'mock_model': True}
        return True
    
    def predict(self, data):
        self._ensure_initialized()
        return {'prediction': 'mock_prediction', 'confidence': 0.9}
    
    def train(self, data):
        self._ensure_initialized()
        return {'status': 'success', 'accuracy': 0.85}


class TestAIService(TestCase):
    """Test cases for the AIService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = MockAIService()
    
    def test_initialization(self):
        """Test that the service can be initialized."""
        # Initially not initialized
        self.assertFalse(self.service.initialized)
        
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(self.service.initialized)
        self.assertIn('mock_model', self.service.models)
    
    def test_get_model_info(self):
        """Test getting model information."""
        # Initialize the service
        self.service.initialize()
        
        # Get model info
        info = self.service.get_model_info()
        
        # Check info structure
        self.assertEqual(info['service_name'], 'MockAIService')
        self.assertTrue(info['initialized'])
        self.assertIn('mock_model', info['models'])
    
    def test_ensure_initialized(self):
        """Test that _ensure_initialized raises an error if not initialized."""
        # Service not initialized
        with self.assertRaises(RuntimeError):
            self.service._ensure_initialized()
        
        # Initialize the service
        self.service.initialize()
        
        # Should not raise an error now
        try:
            self.service._ensure_initialized()
        except RuntimeError:
            self.fail("_ensure_initialized raised RuntimeError unexpectedly!")
    
    def test_predict_requires_initialization(self):
        """Test that predict requires initialization."""
        # Service not initialized
        with self.assertRaises(RuntimeError):
            self.service.predict({'input': 'test'})
        
        # Initialize the service
        self.service.initialize()
        
        # Should work now
        result = self.service.predict({'input': 'test'})
        self.assertEqual(result['prediction'], 'mock_prediction')
    
    def test_train_requires_initialization(self):
        """Test that train requires initialization."""
        # Service not initialized
        with self.assertRaises(RuntimeError):
            self.service.train({'input': 'test'})
        
        # Initialize the service
        self.service.initialize()
        
        # Should work now
        result = self.service.train({'input': 'test'})
        self.assertEqual(result['status'], 'success')
