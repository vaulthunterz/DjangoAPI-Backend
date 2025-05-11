"""
Training and evaluation scripts for expense categorization models.
"""

from .train import train_model
from .evaluate import evaluate_model

__all__ = [
    'train_model',
    'evaluate_model'
]
