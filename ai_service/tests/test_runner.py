"""
Django test runner for AI service tests.
"""
from django.test import TestCase
from django.test.runner import DiscoverRunner

from ai_service.service import AIService
from ai_service.factory import AIServiceFactory
from ai_service.expense_service import ExpenseAIService
from ai_service.investment_service import InvestmentAIService
from ai_service.gemini_service import GeminiAIService


class AIServiceTestCase(TestCase):
    """Base test case for AI service tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset the factory singleton for each test
        AIServiceFactory._instance = None
        AIServiceFactory._services = {}


class AIServiceTestRunner(DiscoverRunner):
    """Custom test runner for AI service tests."""
    
    def run_tests(self, test_labels, extra_tests=None, **kwargs):
        """Run the tests with the AI service test case."""
        if not test_labels:
            test_labels = ['ai_service']
        return super().run_tests(test_labels, extra_tests, **kwargs)
