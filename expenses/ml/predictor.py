import os
import sys
import pandas as pd

# Add the parent directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from expenses.ml.model_training import ExpenseCategoryClassifier
except ImportError:
    from model_training import ExpenseCategoryClassifier

class ExpensePredictor:
    def __init__(self):
        self.classifier = ExpenseCategoryClassifier()
        self.model_loaded = self.classifier.load_model()

    def predict_category(self, transaction_data):
        """
        Predict the category and subcategory for a given transaction.
        
        Args:
            transaction_data (dict): Dictionary containing transaction information
                                   Required keys: 'description', 'merchant'
        
        Returns:
            dict: Predicted category and subcategory with confidence scores
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please train the model first.")

        # Validate required fields
        required_fields = ['description', 'merchant']
        if not all(field in transaction_data for field in required_fields):
            raise ValueError(f"Transaction data must contain: {', '.join(required_fields)}")

        return self.classifier.predict(transaction_data)

    def predict_categories_batch(self, transactions):
        """
        Predict categories for multiple transactions at once.
        
        Args:
            transactions (list): List of transaction dictionaries
                               Each dict must contain 'description' and 'merchant'
        
        Returns:
            list: List of predictions with confidence scores
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please train the model first.")

        # Validate all transactions have required fields
        required_fields = ['description', 'merchant']
        for transaction in transactions:
            if not all(field in transaction for field in required_fields):
                raise ValueError(f"All transactions must contain: {', '.join(required_fields)}")

        results = []
        for transaction in transactions:
            prediction = self.classifier.predict(transaction)
            results.append(prediction)
        
        return results 