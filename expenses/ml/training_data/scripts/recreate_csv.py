"""
Script to recreate the transactions.csv file from scratch with only valid data
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
    
    # Try to read the file with pandas
    try:
        # First try to read with pandas
        df = pd.read_csv(transactions_file, error_bad_lines=False, warn_bad_lines=True)
        print(f"Successfully read {len(df)} rows with pandas")
    except Exception as e:
        print(f"Error reading with pandas: {str(e)}")
        
        # If pandas fails, try a more manual approach with csv module
        print("Attempting manual read with csv module...")
        valid_rows = []
        
        with open(transactions_file, 'r', encoding='utf-8', newline='') as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Get header
            valid_rows.append(header)
            
            for i, row in enumerate(reader, start=1):
                if len(row) == 5:  # Only keep rows with exactly 5 fields
                    valid_rows.append(row)
                else:
                    if i < 20:  # Print first few invalid rows for debugging
                        print(f"Skipping invalid row {i}: {row}")
        
        # Convert to DataFrame
        df = pd.DataFrame(valid_rows[1:], columns=valid_rows[0])
        print(f"Manually read {len(df)} valid rows")
    
    # Clean the data
    print("Cleaning data...")
    
    # Remove any rows with empty values
    original_count = len(df)
    df = df.dropna()
    print(f"Removed {original_count - len(df)} rows with missing values")
    
    # Clean each column
    for col in df.columns:
        # Remove any quotes
        df[col] = df[col].astype(str).str.replace('"', '')
        
        # Remove any newlines or carriage returns
        df[col] = df[col].str.replace('\n', ' ').str.replace('\r', ' ')
        
        # Trim whitespace
        df[col] = df[col].str.strip()
    
    # Write to CSV with proper quoting for merchant column
    print(f"Writing {len(df)} cleaned rows to CSV...")
    
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(df.columns)
        
        # Write data rows
        for _, row in df.iterrows():
            # Convert row to list
            row_list = row.tolist()
            
            # Ensure merchant names with commas are properly quoted
            if ',' in row_list[3]:  # Merchant is the 4th column (index 3)
                row_list[3] = f'"{row_list[3]}"'
            
            writer.writerow(row_list)
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully recreated {transactions_file} with {len(df)} valid rows")
    
    # Verify the file can be read
    try:
        test_df = pd.read_csv(transactions_file)
        print(f"Verification successful: File can be read correctly with {len(test_df)} rows")
    except Exception as e:
        print(f"Verification failed: {str(e)}")

if __name__ == "__main__":
    main()
