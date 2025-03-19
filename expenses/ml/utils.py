import pandas as pd
import csv
import os

def load_training_data(csv_file_path=None):
    """
    Load training data from CSV file or use default training data.
    
    Args:
        csv_file_path (str, optional): Path to CSV file containing training data
        
    Returns:
        pd.DataFrame: DataFrame containing the training data
    """
    if csv_file_path and os.path.exists(csv_file_path):
        return pd.read_csv(csv_file_path)
    
    # Use default training data path
    default_path = os.path.join(
        os.path.dirname(__file__),
        'training_data',
        'transactions.csv'
    )
    
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    
    raise FileNotFoundError("No training data found")

def validate_transaction_data(transaction):
    """
    Validate transaction data format.
    
    Args:
        transaction (dict): Transaction data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_fields = ['merchant', 'amount']
    return all(field in transaction for field in required_fields)

def format_amount(amount):
    """
    Format amount string to numeric value.
    
    Args:
        amount (str/float): Amount to format
        
    Returns:
        float: Formatted amount
    """
    if isinstance(amount, (int, float)):
        return float(amount)
    
    # Remove currency symbols and convert to float
    amount_str = str(amount)
    amount_str = ''.join(c for c in amount_str if c.isdigit() or c in '.-')
    try:
        return float(amount_str)
    except ValueError:
        return 0.0

def get_available_categories():
    """
    Get list of available transaction categories from training data.
    
    Returns:
        list: List of unique categories
    """
    try:
        df = load_training_data()
        return sorted(df['category'].unique().tolist())
    except:
        return []

def save_transaction_to_training_data(transaction, csv_file_path=None):
    """
    Save a new transaction to the training data CSV file.
    
    Args:
        transaction (dict): Transaction data to save
        csv_file_path (str, optional): Path to CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not validate_transaction_data(transaction):
        return False
    
    if csv_file_path is None:
        csv_file_path = os.path.join(
            os.path.dirname(__file__),
            'training_data',
            'transactions.csv'
        )
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file_path)
    
    try:
        mode = 'a' if file_exists else 'w'
        with open(csv_file_path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=transaction.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(transaction)
        return True
    except:
        return False 