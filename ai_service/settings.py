"""
AI Service Settings Module

This module provides configuration settings for the AI services.
"""
import os
from django.conf import settings

# Gemini AI settings
GOOGLE_AI_STUDIO_KEY = getattr(settings, 'GOOGLE_AI_STUDIO_KEY', os.getenv('GOOGLE_AI_STUDIO_KEY', 'AIzaSyB465HZ8X-T5vqTfQuPBo4C_Qh66Q5PZgY'))
GEMINI_MODEL_NAME = getattr(settings, 'GEMINI_MODEL_NAME', os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash'))

# Expense AI settings
EXPENSE_MODEL_PATH = os.path.join('expenses', 'ml')
EXPENSE_CUSTOM_MODEL_PATH = os.path.join(EXPENSE_MODEL_PATH, 'custom_model.joblib')
EXPENSE_VECTORIZER_PATH = os.path.join(EXPENSE_MODEL_PATH, 'vectorizer.joblib')
EXPENSE_LABEL_ENCODER_PATH = os.path.join(EXPENSE_MODEL_PATH, 'label_encoder.joblib')

# Investment AI settings
INVESTMENT_MODEL_PATH = os.path.join('investment', 'ml')
INVESTMENT_RECOMMENDER_PATH = os.path.join(INVESTMENT_MODEL_PATH, 'recommender')

# Logging settings
LOG_LEVEL = getattr(settings, 'AI_SERVICE_LOG_LEVEL', 'INFO')
LOG_FILE = getattr(settings, 'AI_SERVICE_LOG_FILE', None)

# Feature flags
ENABLE_GEMINI_AI = getattr(settings, 'ENABLE_GEMINI_AI', True)
ENABLE_EXPENSE_AI = getattr(settings, 'ENABLE_EXPENSE_AI', True)
ENABLE_INVESTMENT_AI = getattr(settings, 'ENABLE_INVESTMENT_AI', True)

# Performance settings
CACHE_PREDICTIONS = getattr(settings, 'CACHE_AI_PREDICTIONS', True)
CACHE_TIMEOUT = getattr(settings, 'AI_CACHE_TIMEOUT', 3600)  # 1 hour

# Default confidence thresholds
MIN_CONFIDENCE_THRESHOLD = 0.6
