"""
Data utility functions for expense categorization models
"""

import os
import pandas as pd
import csv
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_training_data(data_path=None):
    """Load training data from CSV file"""
    if data_path is None:
        # Use default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(current_dir)
        data_path = os.path.join(ml_dir, 'training_data', 'transactions.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Training data file not found: {data_path}")
        return None
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} transactions from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading training data: {str(e)}")
        return None

def validate_transaction_data(transaction):
    """Validate transaction data for prediction"""
    required_fields = ['description']
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in transaction:
            return False, f"Missing required field: {field}"
    
    # Check if description is not empty
    if not transaction['description']:
        return False, "Description cannot be empty"
    
    return True, "Transaction data is valid"

def format_amount(amount):
    """Format amount as a string with 2 decimal places"""
    try:
        return f"{float(amount):.2f}"
    except (ValueError, TypeError):
        return "0.00"

def get_available_categories(include_subcategories=False):
    """Get available categories and subcategories from training data"""
    df = load_training_data()
    if df is None:
        return []
    
    if not include_subcategories:
        return sorted(df['category'].unique().tolist())
    
    # Include subcategories
    result = {}
    for category in sorted(df['category'].unique()):
        subcategories = sorted(df[df['category'] == category]['subcategory'].unique().tolist())
        result[category] = subcategories
    
    return result

def save_transaction_to_training_data(transaction, data_path=None):
    """Save a transaction to the training data CSV file"""
    if data_path is None:
        # Use default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.dirname(current_dir)
        data_path = os.path.join(ml_dir, 'training_data', 'transactions.csv')
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Check if file exists
    file_exists = os.path.exists(data_path)
    
    # Prepare row data
    row = {
        'description': transaction.get('description', ''),
        'merchant': transaction.get('merchant', ''),
        'amount': format_amount(transaction.get('amount', 0)),
        'category': transaction.get('category', ''),
        'subcategory': transaction.get('subcategory', ''),
        'is_expense': transaction.get('is_expense', 1)
    }
    
    # Write to CSV
    try:
        with open(data_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
        
        logger.info(f"Transaction saved to {data_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving transaction: {str(e)}")
        return False
