"""
Test runner for AI service tests.
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import the ai_service module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test modules
from test_service import TestAIService
from test_factory import TestAIServiceFactory
from test_expense_service import TestExpenseAIService
from test_investment_service import TestInvestmentAIService
from test_gemini_service import TestGeminiAIService


def run_tests():
    """Run all AI service tests."""
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestAIService))
    test_suite.addTest(unittest.makeSuite(TestAIServiceFactory))
    test_suite.addTest(unittest.makeSuite(TestExpenseAIService))
    test_suite.addTest(unittest.makeSuite(TestInvestmentAIService))
    test_suite.addTest(unittest.makeSuite(TestGeminiAIService))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return the result
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(not result.wasSuccessful())
