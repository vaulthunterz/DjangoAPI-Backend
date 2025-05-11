"""
Configuration settings for machine learning models.

This file contains all configurable parameters for the ML models,
making it easy to adjust settings in one central location.
"""

import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# ======================================================================
# Feature Extraction Settings
# ======================================================================

# Feature weights for text columns
# Set weight to 0.0 to completely ignore a column
FEATURE_WEIGHTS = {
    'description': 10.0,  # Weight for description field
    'merchant': 0.0,      # Weight for merchant field (0.0 = ignore)
}

# Feature extraction parameters
MAX_FEATURES = 500        # Maximum number of features to extract per column
NGRAM_RANGE = (1, 3)      # Range of n-grams to extract (1=unigrams, 2=bigrams, 3=trigrams)

# ======================================================================
# Model Training Settings
# ======================================================================

# Test size for train/test split
TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# Whether to use ensemble model (True) or just RandomForest (False)
USE_ENSEMBLE = True

# Number of parallel jobs to use (-1 means use all available cores)
N_JOBS = -1

# ======================================================================
# RandomForest Classifier Settings (Category Model)
# ======================================================================

RF_CATEGORY_PARAMS = {
    'n_estimators': 100,      # Number of trees in the forest
    'max_depth': 20,          # Maximum depth of the trees
    'min_samples_split': 5,   # Minimum samples required to split a node
    'min_samples_leaf': 2,    # Minimum samples required at a leaf node
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,         # Number of parallel jobs
    'class_weight': None,     # Weight for imbalanced classes (None, 'balanced', or dict)
    'verbose': 0              # Verbosity level (0 = silent, higher = more verbose)
}

# ======================================================================
# GradientBoosting Classifier Settings (Category Model)
# ======================================================================

GB_CATEGORY_PARAMS = {
    'n_estimators': 100,      # Number of boosting stages
    'learning_rate': 0.1,     # Learning rate shrinks the contribution of each tree
    'max_depth': 5,           # Maximum depth of the individual regression estimators
    'random_state': RANDOM_STATE,
    'verbose': 1,             # Verbosity level (0 = silent, higher = more verbose)
    'subsample': 1.0,         # Fraction of samples used for fitting the individual trees
    'min_samples_split': 2,   # Minimum samples required to split a node
    'min_samples_leaf': 1     # Minimum samples required at a leaf node
}

# ======================================================================
# LogisticRegression Classifier Settings (Category Model)
# ======================================================================

LR_CATEGORY_PARAMS = {
    'C': 1.0,                 # Inverse of regularization strength
    'max_iter': 1000,         # Maximum number of iterations
    'multi_class': 'multinomial',  # Type of multiclass approach
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,         # Number of parallel jobs
    'solver': 'lbfgs',        # Algorithm to use in the optimization problem
    'class_weight': None      # Weight for imbalanced classes (None, 'balanced', or dict)
}

# ======================================================================
# RandomForest Classifier Settings (Subcategory Model)
# ======================================================================

RF_SUBCATEGORY_PARAMS = {
    'n_estimators': 50,       # Number of trees in the forest
    'max_depth': 15,          # Maximum depth of the trees
    'min_samples_split': 4,   # Minimum samples required to split a node
    'min_samples_leaf': 2,    # Minimum samples required at a leaf node
    'random_state': RANDOM_STATE,
    'n_jobs': N_JOBS,         # Number of parallel jobs
    'class_weight': None,     # Weight for imbalanced classes (None, 'balanced', or dict)
    'verbose': 0              # Verbosity level (0 = silent, higher = more verbose)
}

# ======================================================================
# GradientBoosting Classifier Settings (Subcategory Model)
# ======================================================================

GB_SUBCATEGORY_PARAMS = {
    'n_estimators': 50,       # Number of boosting stages
    'learning_rate': 0.1,     # Learning rate shrinks the contribution of each tree
    'max_depth': 4,           # Maximum depth of the individual regression estimators
    'random_state': RANDOM_STATE,
    'verbose': 0,             # Verbosity level (0 = silent, higher = more verbose)
    'subsample': 1.0,         # Fraction of samples used for fitting the individual trees
    'min_samples_split': 2,   # Minimum samples required to split a node
    'min_samples_leaf': 1     # Minimum samples required at a leaf node
}

# ======================================================================
# Ensemble Model Settings
# ======================================================================

# Weights for each model in the category ensemble
CATEGORY_ENSEMBLE_WEIGHTS = {
    'random_forest': 0.5,     # Weight for RandomForest model
    'gradient_boosting': 0.3, # Weight for GradientBoosting model
    'logistic_regression': 0.2  # Weight for LogisticRegression model
}

# Weights for each model in the subcategory ensemble
SUBCATEGORY_ENSEMBLE_WEIGHTS = {
    'random_forest': 0.6,     # Weight for RandomForest model
    'gradient_boosting': 0.4  # Weight for GradientBoosting model
}

# ======================================================================
# Model Evaluation Settings
# ======================================================================

# Metrics to calculate during model evaluation
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1'
]

# Number of top features to display in feature importance
TOP_FEATURES_COUNT = 20

# ======================================================================
# Model Paths
# ======================================================================

# Base directory for models
BASE_MODELS_DIR = 'models'

# Function to get the full path to the models directory
def get_models_dir(base_dir=None):
    """
    Get the full path to the models directory.

    Args:
        base_dir: Optional base directory to use instead of the default

    Returns:
        str: Full path to the models directory
    """
    if base_dir:
        models_dir = os.path.join(base_dir, BASE_MODELS_DIR)
    else:
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, BASE_MODELS_DIR)

    # Create the directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir

# ======================================================================
# Utility Functions
# ======================================================================

def log_config_settings():
    """Log the current configuration settings."""
    logger.info("=" * 80)
    logger.info("ML CONFIGURATION SETTINGS")
    logger.info("=" * 80)

    # Feature extraction settings
    logger.info("Feature Extraction Settings:")
    logger.info(f"  Feature Weights: description={FEATURE_WEIGHTS['description']}, merchant={FEATURE_WEIGHTS['merchant']}")
    logger.info(f"  Max Features: {MAX_FEATURES}")
    logger.info(f"  N-gram Range: {NGRAM_RANGE}")

    # Model training settings
    logger.info("Model Training Settings:")
    logger.info(f"  Test Size: {TEST_SIZE}")
    logger.info(f"  Random State: {RANDOM_STATE}")
    logger.info(f"  Use Ensemble: {USE_ENSEMBLE}")
    logger.info(f"  Number of Jobs: {N_JOBS}")

    # RandomForest settings
    logger.info("RandomForest Category Model Settings:")
    for param, value in RF_CATEGORY_PARAMS.items():
        logger.info(f"  {param}: {value}")

    # GradientBoosting settings
    logger.info("GradientBoosting Category Model Settings:")
    for param, value in GB_CATEGORY_PARAMS.items():
        logger.info(f"  {param}: {value}")

    # LogisticRegression settings
    logger.info("LogisticRegression Category Model Settings:")
    for param, value in LR_CATEGORY_PARAMS.items():
        logger.info(f"  {param}: {value}")

    # Subcategory model settings
    logger.info("RandomForest Subcategory Model Settings:")
    for param, value in RF_SUBCATEGORY_PARAMS.items():
        logger.info(f"  {param}: {value}")

    logger.info("GradientBoosting Subcategory Model Settings:")
    for param, value in GB_SUBCATEGORY_PARAMS.items():
        logger.info(f"  {param}: {value}")

    # Ensemble weights
    logger.info("Ensemble Weights:")
    logger.info(f"  Category Model: {CATEGORY_ENSEMBLE_WEIGHTS}")
    logger.info(f"  Subcategory Model: {SUBCATEGORY_ENSEMBLE_WEIGHTS}")

    # Model paths
    logger.info("Model Paths:")
    logger.info(f"  Models Directory: {get_models_dir()}")

    logger.info("=" * 80)

def update_feature_weights(description_weight=None, merchant_weight=None):
    """
    Update the feature weights.

    Args:
        description_weight: New weight for description field
        merchant_weight: New weight for merchant field

    Returns:
        dict: Updated feature weights
    """
    global FEATURE_WEIGHTS

    if description_weight is not None:
        FEATURE_WEIGHTS['description'] = float(description_weight)

    if merchant_weight is not None:
        FEATURE_WEIGHTS['merchant'] = float(merchant_weight)

    logger.info(f"Updated feature weights: description={FEATURE_WEIGHTS['description']}, merchant={FEATURE_WEIGHTS['merchant']}")

    return FEATURE_WEIGHTS
