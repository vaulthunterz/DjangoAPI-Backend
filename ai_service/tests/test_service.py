"""
Tests for the base AIService class.
"""
import unittest
from unittest.mock import patch, MagicMock

from ai_service.service import AIService


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


class TestAIService(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
