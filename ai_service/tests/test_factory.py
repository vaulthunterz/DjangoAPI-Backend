"""
Tests for the AIServiceFactory class.
"""
import unittest
from unittest.mock import patch, MagicMock

from ai_service.factory import AIServiceFactory
from ai_service.service import AIService
from ai_service.expense_service import ExpenseAIService
from ai_service.investment_service import InvestmentAIService
from ai_service.gemini_service import GeminiAIService


class TestAIServiceFactory(unittest.TestCase):
    """Test cases for the AIServiceFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a new factory instance for each test
        self.factory = AIServiceFactory()
        
        # Reset the singleton instance and services
        AIServiceFactory._instance = None
        AIServiceFactory._services = {}
    
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
    
    @patch('ai_service.factory.AIServiceFactory.get_service')
    def test_get_all_services(self, mock_get_service):
        """Test getting all services."""
        # Mock the get_service method
        mock_get_service.side_effect = lambda service_type: MagicMock(spec=AIService)
        
        # Get all services
        services = self.factory.get_all_services()
        
        # Check that get_service was called for each service type
        self.assertEqual(mock_get_service.call_count, 3)
        
        # Check that we got all three services
        self.assertEqual(len(services), 3)
    
    @patch('ai_service.factory.AIServiceFactory.get_service')
    def test_get_service_info(self, mock_get_service):
        """Test getting service info."""
        # Create mock services
        mock_expense = MagicMock(spec=AIService)
        mock_expense.get_model_info.return_value = {'service_name': 'ExpenseAIService'}
        
        mock_investment = MagicMock(spec=AIService)
        mock_investment.get_model_info.return_value = {'service_name': 'InvestmentAIService'}
        
        mock_gemini = MagicMock(spec=AIService)
        mock_gemini.get_model_info.return_value = {'service_name': 'GeminiAIService'}
        
        # Mock the get_service method
        mock_get_service.side_effect = lambda service_type: {
            'expense': mock_expense,
            'investment': mock_investment,
            'gemini': mock_gemini
        }[service_type]
        
        # Get service info
        info = self.factory.get_service_info()
        
        # Check that we got info for all three services
        self.assertEqual(len(info), 3)
        self.assertEqual(info['expense']['service_name'], 'ExpenseAIService')
        self.assertEqual(info['investment']['service_name'], 'InvestmentAIService')
        self.assertEqual(info['gemini']['service_name'], 'GeminiAIService')


if __name__ == '__main__':
    unittest.main()
