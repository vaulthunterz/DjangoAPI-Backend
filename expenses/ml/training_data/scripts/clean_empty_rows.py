"""
Script to clean the transactions.csv file by removing empty rows and ensuring proper formatting
"""

import os
import csv
import pandas as pd
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
    
    # Read the file line by line
    valid_rows = []
    with open(transactions_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Process header
    header = lines[0].strip()
    header_parts = header.split(',')
    valid_rows.append(header)
    
    # Process data rows
    empty_rows = 0
    invalid_rows = 0
    
    for i, line in enumerate(lines[1:], start=1):
        line = line.strip()
        
        # Skip completely empty lines
        if not line:
            empty_rows += 1
            continue
        
        # Skip lines that are just commas
        if line.replace(',', '').strip() == '':
            empty_rows += 1
            continue
        
        # Split by commas
        parts = line.split(',')
        
        # Check if we have data in all required fields
        if len(parts) >= 5 and all(parts[:3]) and parts[-1]:  # Check first 3 fields and amount
            # This is a valid row
            valid_rows.append(line)
        else:
            invalid_rows += 1
            if i <= 10:  # Print first few invalid rows for debugging
                print(f"Skipping invalid row {i}: {line}")
    
    print(f"Found {empty_rows} empty rows and {invalid_rows} invalid rows")
    
    # Write the valid rows to the file
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('\n'.join(valid_rows))
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully cleaned CSV file: {transactions_file}")
    print(f"Kept {len(valid_rows)-1} valid data rows")
    
    # Verify the file can be read by pandas
    try:
        df = pd.read_csv(transactions_file)
        print(f"Verification successful: File can be read correctly with {len(df)} rows")
        
        # Check for any missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"Warning: Found {missing_values} missing values in the data")
        else:
            print("No missing values found in the data")
        
        # Check column counts
        print("Column counts:")
        for col in df.columns:
            print(f"  {col}: {df[col].count()}")
    except Exception as e:
        print(f"Verification failed: {str(e)}")

if __name__ == "__main__":
    main()
