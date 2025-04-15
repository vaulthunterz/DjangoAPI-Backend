# AI Service Tests

This directory contains unit tests for the AI service module.

## Test Structure

- `test_simple.py`: Standalone tests that don't require Django settings
- `test_django.py`: Tests that integrate with Django's test framework
- `test_service.py`: Tests for the base `AIService` class
- `test_factory.py`: Tests for the `AIServiceFactory` class
- `test_expense_service.py`: Tests for the `ExpenseAIService` class
- `test_investment_service.py`: Tests for the `InvestmentAIService` class
- `test_gemini_service.py`: Tests for the `GeminiAIService` class
- `run_tests.py`: Script to run all tests
- `test_runner.py`: Django test runner integration

## Running Tests

### Using the standalone test runner for simple tests

```bash
cd backend
python -m ai_service.tests.test_simple
```

### Using Django's test framework

```bash
cd backend
python manage.py test ai_service
```

## Test Coverage

The tests cover the following aspects of the AI service:

- Initialization and configuration
- Prediction functionality
- Training functionality
- Error handling
- Factory pattern implementation
- Service-specific functionality

## Mocking

The tests use mocking to isolate the AI service from external dependencies:

- External APIs (Google Gemini)
- Django models
- File system operations
- Machine learning models

## Test Environment Setup

### Option 1: Using .env file (Recommended)

The Django settings module is now configured in the .env file, so you only need to:

1. Activate the virtual environment:
   ```bash
   & "C:\Users\Bedan\PycharmProjects\1 Python\ExpeCatAPI\.venv\Scripts\Activate.ps1"
   ```

2. Run the tests:
   ```bash
   python -m unittest discover -s ai_service/tests -p "test_*.py"
   ```

   Or using Django's test runner:
   ```bash
   python manage.py test ai_service
   ```

### Option 2: Setting environment variable manually

If the .env file is not properly loaded, you can set the environment variable manually:

1. Activate the virtual environment:
   ```bash
   & "C:\Users\Bedan\PycharmProjects\1 Python\ExpeCatAPI\.venv\Scripts\Activate.ps1"
   ```

2. Set the Django settings module:
   ```bash
   $env:DJANGO_SETTINGS_MODULE="ExpenseCategorizationAPI.settings"
   ```

3. Run the tests:
   ```bash
   python -m unittest discover -s ai_service/tests -p "test_*.py"
   ```

## Troubleshooting

If you encounter the error `ImproperlyConfigured: Requested setting INSTALLED_APPS, but settings are not configured`, make sure you've set the `DJANGO_SETTINGS_MODULE` environment variable correctly.

For PowerShell:
```powershell
$env:DJANGO_SETTINGS_MODULE="ExpenseCategorizationAPI.settings"
```

For Command Prompt:
```cmd
set DJANGO_SETTINGS_MODULE=ExpenseCategorizationAPI.settings
```

For Bash:
```bash
export DJANGO_SETTINGS_MODULE=ExpenseCategorizationAPI.settings
```

## Adding New Tests

When adding new functionality to the AI service, please add corresponding tests to maintain test coverage.
