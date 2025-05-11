"""
Expense categorization machine learning module
"""

# Import core model
from .core.model import ExpenseCategorizer
from .core.features import create_enhanced_features

# Import utility functions
from .utils.data_utils import (
    load_training_data,
    validate_transaction_data,
    format_amount,
    get_available_categories,
    save_transaction_to_training_data
)

from .utils.model_utils import (
    save_performance_metrics,
    print_performance_summary
)

# Import scripts
from .scripts.train import train_model
from .scripts.evaluate import evaluate_model

__all__ = [
    # Core
    'ExpenseCategorizer',
    'create_enhanced_features',

    # Utils
    'load_training_data',
    'validate_transaction_data',
    'format_amount',
    'get_available_categories',
    'save_transaction_to_training_data',
    'save_performance_metrics',
    'print_performance_summary',

    # Scripts
    'train_model',
    'evaluate_model'
]