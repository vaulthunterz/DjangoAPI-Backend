"""
Simple tests for the AI service module.
"""
import unittest
from unittest.mock import patch, MagicMock


class MockAIService:
    """Mock AI service for testing."""
    
    def __init__(self):
        self.initialized = False
        self.models = {}
    
    def initialize(self):
        self.initialized = True
        self.models = {'mock_model': True}
        return True
    
    def predict(self, data):
        if not self.initialized:
            raise RuntimeError("Service not initialized")
        return {'prediction': 'mock_prediction', 'confidence': 0.9}
    
    def train(self, data):
        if not self.initialized:
            raise RuntimeError("Service not initialized")
        return {'status': 'success', 'accuracy': 0.85}
    
    def get_model_info(self):
        return {
            'service_name': 'MockAIService',
            'initialized': self.initialized,
            'models': list(self.models.keys())
        }


class MockServiceFactory:
    """Mock service factory for testing."""
    
    _instance = None
    _services = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MockServiceFactory, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._services = {}
            self._initialized = True
    
    def get_service(self, service_type):
        if service_type in self._services:
            return self._services[service_type]
        
        if service_type == 'mock':
            service = MockAIService()
            service.initialize()
            self._services[service_type] = service
            return service
        
        raise ValueError(f"Unknown service type: {service_type}")
    
    def get_all_services(self):
        return self._services
    
    def get_service_info(self):
        info = {}
        for service_type, service in self._services.items():
            info[service_type] = service.get_model_info()
        return info


class TestMockAIService(unittest.TestCase):
    """Test cases for the mock AI service."""
    
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


class TestMockServiceFactory(unittest.TestCase):
    """Test cases for the mock service factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the singleton instance and services
        MockServiceFactory._instance = None
        MockServiceFactory._services = {}
        
        # Create a new factory instance for each test
        self.factory = MockServiceFactory()
    
    def test_singleton_pattern(self):
        """Test that the factory follows the singleton pattern."""
        # Create two factory instances
        factory1 = MockServiceFactory()
        factory2 = MockServiceFactory()
        
        # They should be the same instance
        self.assertIs(factory1, factory2)
    
    def test_get_service(self):
        """Test getting a service."""
        # Get the mock service
        service = self.factory.get_service('mock')
        
        # Check that we got a service
        self.assertIsInstance(service, MockAIService)
        
        # Get the service again
        service2 = self.factory.get_service('mock')
        
        # Should be the same instance
        self.assertIs(service, service2)
    
    def test_get_unknown_service(self):
        """Test getting an unknown service type."""
        # Try to get an unknown service
        with self.assertRaises(ValueError):
            self.factory.get_service('unknown')
    
    def test_get_service_info(self):
        """Test getting service info."""
        # Get a service to populate the factory
        self.factory.get_service('mock')
        
        # Get service info
        info = self.factory.get_service_info()
        
        # Check that we got info for the mock service
        self.assertIn('mock', info)
        self.assertEqual(info['mock']['service_name'], 'MockAIService')


if __name__ == '__main__':
    unittest.main()
