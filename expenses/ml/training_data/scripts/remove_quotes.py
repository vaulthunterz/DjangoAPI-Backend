"""
Script to remove quotes from the description column in transactions.csv
"""

import os
import pandas as pd
import re
from datetime import datetime
import shutil

def backup_file(file_path):
    """Create a backup of the file."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of original file at {backup_path}")
        return True
    return False

def remove_quotes(text):
    """
    Remove all quotes from text and handle CSV escaping properly.
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove double quotes
    text = text.replace('"', '')
    
    # Remove single quotes
    text = text.replace("'", "")
    
    # Standardize whitespace
    text = ' '.join(text.split())
    
    return text

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    transactions_file = os.path.join(base_dir, 'transactions.csv')
    
    # Backup the original file
    backup_file(transactions_file)
    
    # Read the transactions file
    print("Reading transactions file...")
    
    # Read with pandas, but be careful with quote handling
    df = pd.read_csv(transactions_file, quotechar='"', escapechar='\\')
    
    # Get initial count
    initial_count = len(df)
    print(f"Initial number of transactions: {initial_count}")
    
    # Count entries with quotes in description
    quotes_count = df[df['description'].str.contains('"') | df['description'].str.contains("'")].shape[0]
    print(f"Found {quotes_count} transactions with quotes in description")
    
    # Clean descriptions by removing quotes
    print("Removing quotes from description text...")
    df['description'] = df['description'].apply(remove_quotes)
    
    # Save the cleaned data
    # Use quoting=csv.QUOTE_MINIMAL to only quote fields when necessary
    df.to_csv(transactions_file, index=False, quoting=1)
    print(f"Successfully saved {len(df)} cleaned transactions to {transactions_file}")
    print(f"Removed quotes from all descriptions")

if __name__ == "__main__":
    main()
