"""
Detailed test runner for AI service tests.

This script runs all the tests in the AI service module and provides detailed output
for each test case, including what was tested and whether it passed or failed.
"""
import unittest
import sys
import os
import time
from unittest.mock import patch, MagicMock

# Import test modules directly to avoid Django settings issues
from test_simple import TestMockAIService, TestMockServiceFactory


class DetailedTestResult(unittest.TextTestResult):
    """Custom test result class that provides detailed output for each test."""

    def startTest(self, test):
        """Called when a test begins."""
        super().startTest(test)
        test_name = test._testMethodName
        test_doc = test._testMethodDoc or "No description available"
        print(f"\n{'='*80}")
        print(f"RUNNING: {test.__class__.__name__}.{test_name}")
        print(f"DESCRIPTION: {test_doc}")
        print(f"{'-'*80}")
        self._start_time = time.time()

    def addSuccess(self, test):
        """Called when a test succeeds."""
        super().addSuccess(test)
        elapsed = time.time() - self._start_time
        print(f"{'-'*80}")
        print(f"RESULT: PASS (took {elapsed:.3f}s)")
        print(f"{'='*80}")

    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        elapsed = time.time() - self._start_time
        print(f"{'-'*80}")
        print(f"RESULT: FAIL (took {elapsed:.3f}s)")
        print(f"ERROR: {err[1]}")
        print(f"{'='*80}")

    def addError(self, test, err):
        """Called when a test raises an error."""
        super().addError(test, err)
        elapsed = time.time() - self._start_time
        print(f"{'-'*80}")
        print(f"RESULT: ERROR (took {elapsed:.3f}s)")
        print(f"ERROR: {err[1]}")
        print(f"{'='*80}")


class DetailedTestRunner(unittest.TextTestRunner):
    """Custom test runner that uses the DetailedTestResult class."""

    def __init__(self, *args, **kwargs):
        """Initialize the test runner."""
        kwargs['resultclass'] = DetailedTestResult
        super().__init__(*args, **kwargs)


def run_detailed_tests():
    """Run all AI service tests with detailed output."""
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    for test_case in unittest.defaultTestLoader.loadTestsFromTestCase(TestMockAIService):
        test_suite.addTest(test_case)

    for test_case in unittest.defaultTestLoader.loadTestsFromTestCase(TestMockServiceFactory):
        test_suite.addTest(test_case)

    # Run the tests
    runner = DetailedTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return the result
    return result


if __name__ == '__main__':
    print("\n" + "="*80)
    print("RUNNING DETAILED TESTS FOR AI SERVICE")
    print("="*80 + "\n")

    result = run_detailed_tests()

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {result.testsRun} tests run, {len(result.errors)} errors, {len(result.failures)} failures")
    print("="*80 + "\n")

    sys.exit(not result.wasSuccessful())
