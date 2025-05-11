"""
Script to clean the transactions.csv file:
1. Remove all entries with "Uncategorized" in the merchant column
2. Clean up punctuation symbols and emojis in the description column
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

def clean_description(text):
    """
    Clean description text by:
    1. Removing emojis
    2. Removing quotes
    3. Removing trailing periods
    4. Standardizing whitespace
    """
    # Convert to string if not already
    text = str(text)
    
    # Remove emojis (Unicode ranges for most common emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)
    
    # Remove quotes
    text = text.replace('"', '').replace("'", "")
    
    # Remove trailing periods
    text = text.rstrip('.')
    
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
    df = pd.read_csv(transactions_file)
    
    # Get initial counts
    initial_count = len(df)
    print(f"Initial number of transactions: {initial_count}")
    
    # Remove entries with "Uncategorized" in the merchant column
    uncategorized_count = df[df['merchant'].str.contains('Uncategorized', case=False, na=False)].shape[0]
    df = df[~df['merchant'].str.contains('Uncategorized', case=False, na=False)]
    print(f"Removed {uncategorized_count} transactions with 'Uncategorized' merchant")
    
    # Clean descriptions
    print("Cleaning description text...")
    df['description'] = df['description'].apply(clean_description)
    
    # Save the cleaned data
    df.to_csv(transactions_file, index=False)
    print(f"Successfully saved {len(df)} cleaned transactions to {transactions_file}")
    print(f"Removed a total of {initial_count - len(df)} transactions")

if __name__ == "__main__":
    main()
