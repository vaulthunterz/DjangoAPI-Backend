# AI Service Module

This module provides a centralized interface for all AI-related functionality in the application.

## Features

- Centralized AI service for expense categorization, investment recommendations, and chatbot functionality
- Singleton pattern for efficient resource usage
- Consistent error handling and logging
- Configurable through Django settings

## Usage

### Expense AI Service

```python
from ai_service import get_expense_ai_service

# Get the expense AI service
expense_service = get_expense_ai_service()

# Predict category for a transaction
prediction = expense_service.predict({
    'description': 'Grocery shopping at Walmart',
    'merchant': 'Walmart'
})

# Train a custom model
training_results = expense_service.train_custom_model(transactions)
```

### Investment AI Service

```python
from ai_service import get_investment_ai_service

# Get the investment AI service
investment_service = get_investment_ai_service()

# Generate investment recommendations
recommendations = investment_service.predict(user_profile)

# Analyze a portfolio
analysis = investment_service.analyze_portfolio(portfolio)
```

### Gemini AI Service

```python
from ai_service import get_gemini_ai_service

# Get the Gemini AI service
gemini_service = get_gemini_ai_service()

# Generate a chatbot response
response = gemini_service.get_chatbot_response("How can I save money?")

# Predict a category
prediction = gemini_service.predict_category("Grocery shopping at Walmart", categories)
```

## Configuration

The AI service can be configured through Django settings:

```python
# settings.py

# Gemini AI settings
GOOGLE_AI_STUDIO_KEY = 'your-api-key'
GEMINI_MODEL_NAME = 'gemini-2.0-flash'

# Feature flags
ENABLE_GEMINI_AI = True
ENABLE_EXPENSE_AI = True
ENABLE_INVESTMENT_AI = True
```

## Testing

The AI service module includes comprehensive unit tests. For more information, see the [tests README](tests/README.md).

### Running Tests

```bash
# Activate the virtual environment
& "C:\Users\Bedan\PycharmProjects\1 Python\ExpeCatAPI\.venv\Scripts\Activate.ps1"

# Run all tests
python -m unittest discover -s ai_service/tests -p "test_*.py"

# Or using Django's test runner
python manage.py test ai_service
```
