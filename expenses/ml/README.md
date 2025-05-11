# Expense Categorization ML Module

This module provides machine learning functionality for categorizing expenses based on transaction descriptions, merchants, and amounts.

## Directory Structure

```
ml/
├── core/                  # Core model implementation
│   ├── model.py           # Main ExpenseCategorizer class
│   └── features.py        # Feature extraction functionality
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data handling utilities
│   └── model_utils.py     # Model utilities
├── scripts/               # Training and evaluation scripts
│   ├── train.py           # Model training script
│   └── evaluate.py        # Model evaluation script
├── data/                  # Data storage
├── models/                # Trained models
└── training_data/         # Training data
```

## Features

- **Enhanced Feature Extraction**: Extracts meaningful features from transaction descriptions and merchant names
- **Ensemble Modeling**: Combines multiple models for better accuracy
- **Hierarchical Classification**: Predicts both category and subcategory
- **Visual Progress Tracking**: Shows detailed progress during model training
- **Customizable Parameters**: Easily adjust model parameters for your specific needs

## Requirements

- Python 3.8+
- Required packages: pandas, numpy, scikit-learn, nltk, tqdm, matplotlib, seaborn, tabulate

## Installation

Install the required dependencies:

```bash
# Install required packages
pip install pandas numpy scikit-learn nltk matplotlib seaborn tabulate tqdm

# Download NLTK data
python -m nltk.downloader stopwords wordnet punkt
```

## Usage

### Training a Model

```python
from expenses.ml import train_model

# Train with default settings
categorizer, results = train_model()

# Train with custom settings
categorizer, results = train_model(
    data_path='custom_data.csv',
    models_dir='custom_models',
    sample_size=1000,  # Use a sample of the data
    test_size=0.3,     # Use 30% for testing
    random_state=42    # For reproducibility
)
```

### Evaluating a Model

```python
from expenses.ml import evaluate_model

# Evaluate with default settings
results = evaluate_model()

# Evaluate with custom settings
results = evaluate_model(
    data_path='test_data.csv',
    models_dir='custom_models',
    output_dir='evaluation_results'
)
```

### Making Predictions

```python
from expenses.ml import ExpenseCategorizer

# Initialize and load a trained model
categorizer = ExpenseCategorizer(models_dir='models')
categorizer.load_models()

# Make a prediction
transaction = {
    'description': 'Domain renewal',
    'merchant': 'iBiz Africa',
    'amount': 464.5
}

prediction = categorizer.predict(transaction)
print(f"Predicted category: {prediction['category']}")
print(f"Predicted subcategory: {prediction['subcategory']}")
```

## Model Architecture

The model uses a multi-stage approach:

1. **Feature Extraction**:
   - Text preprocessing (lowercasing, tokenization, etc.)
   - TF-IDF vectorization of description and merchant text
   - Custom feature engineering (text length, keyword presence, etc.)

2. **Category Classification**:
   - Ensemble of RandomForest, GradientBoosting, and LogisticRegression models
   - Weighted voting for final prediction

3. **Subcategory Classification**:
   - Category-specific models for subcategory prediction
   - Ensemble of RandomForest and GradientBoosting models

## Data Format

The training data should be a CSV file with the following columns:
- `description`: Transaction description (required)
- `merchant`: Merchant name (optional)
- `amount`: Transaction amount (optional)
- `category`: Expense category (required for training)
- `subcategory`: Expense subcategory (optional for training)
- `is_expense`: Whether the transaction is an expense (1) or income (0) (optional)

## Performance Metrics

The model provides detailed performance metrics:
- Category accuracy
- Subcategory accuracy by category
- Feature importance
- Training time

## Troubleshooting

If you encounter issues:

1. **Memory Errors**: Reduce sample size or feature count
2. **Slow Training**: Use a smaller dataset for testing
3. **Low Accuracy**: Add more training examples, especially for rare categories
4. **Import Errors**: Check that all dependencies are installed
