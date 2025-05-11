"""
Script to remove all quotes from the transactions.csv file
This script reads and processes the file as raw text to ensure all quotes are removed
"""

import os
import csv
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

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    transactions_file = os.path.join(base_dir, 'transactions.csv')
    temp_file = os.path.join(base_dir, 'transactions_temp.csv')
    
    # Backup the original file
    backup_file(transactions_file)
    
    print("Reading transactions file...")
    
    # Read the original CSV file and write to a new file without quotes
    with open(transactions_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # Read the entire file content
        content = infile.read()
        
        # Count quotes before removal
        quote_count = content.count('"')
        print(f"Found {quote_count} quote characters in the file")
        
        # Remove all double quotes
        content = content.replace('"', '')
        
        # Write the modified content to the temporary file
        outfile.write(content)
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully removed all quotes from {transactions_file}")
    print(f"Removed {quote_count} quote characters")

if __name__ == "__main__":
    main()
