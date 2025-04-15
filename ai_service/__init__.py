# Use direct imports for base classes and factory functions
from .service import AIService
from .factory import (
    AIServiceFactory,
    ai_service_factory,
    get_expense_ai_service,
    get_investment_ai_service,
    get_gemini_ai_service
)

# These will be imported when needed to avoid circular imports
# during Django app initialization
def _import_service_classes():
    from .expense_service import ExpenseAIService
    from .investment_service import InvestmentAIService
    from .gemini_service import GeminiAIService
    return ExpenseAIService, InvestmentAIService, GeminiAIService

# Define what's available when importing from this package
__all__ = [
    'AIService',
    'AIServiceFactory',
    'ai_service_factory',
    'get_expense_ai_service',
    'get_investment_ai_service',
    'get_gemini_ai_service'
]

# Add service classes to __all__ dynamically to avoid import errors
try:
    ExpenseAIService, InvestmentAIService, GeminiAIService = _import_service_classes()
    __all__.extend(['ExpenseAIService', 'InvestmentAIService', 'GeminiAIService'])
except ImportError:
    # During Django initialization, some imports might not be available yet
    pass
