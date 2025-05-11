"""
Script to fix CSV format issues in transactions.csv by properly quoting all fields
This ensures merchant names with commas are handled correctly
"""

import os
import csv
from datetime import datetime
import shutil
import pandas as pd

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
    
    # First, try to read the file line by line to identify problematic lines
    try:
        # Try to read the file with pandas, which might fail
        df = pd.read_csv(transactions_file)
        print(f"Successfully read {len(df)} rows with pandas")
    except Exception as e:
        print(f"Error reading with pandas: {str(e)}")
        
        # If pandas fails, try a more manual approach
        print("Attempting manual fix...")
        
        # Read the file as text
        with open(transactions_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        # Parse each line manually
        data = []
        header = lines[0].strip().split(',')
        data.append(header)
        
        for i, line in enumerate(lines[1:], start=1):
            # Split by commas, but be smarter about it
            parts = []
            current_part = ""
            in_quotes = False
            
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                    current_part += char
                elif char == ',' and not in_quotes:
                    parts.append(current_part)
                    current_part = ""
                else:
                    current_part += char
            
            # Add the last part
            if current_part:
                parts.append(current_part)
            
            # If we don't have exactly 5 parts, try to fix it
            if len(parts) != 5:
                print(f"Line {i+1} has {len(parts)} parts instead of 5: {line.strip()}")
                
                # Try to reconstruct the line based on expected format
                # This is a simplified approach - we assume the last 3 fields are correct
                if len(parts) > 5:
                    # Combine extra fields into the description
                    description = ','.join(parts[:-4])
                    category = parts[-4]
                    subcategory = parts[-3]
                    merchant = parts[-2]
                    amount = parts[-1]
                    parts = [description, category, subcategory, merchant, amount]
            
            data.append(parts)
        
        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])
    
    # Now write the data back with proper quoting
    print("Writing data with proper quoting...")
    
    # Make sure all columns are treated as strings
    for col in df.columns:
        df[col] = df[col].astype(str)
    
    # Write to CSV with quoting
    df.to_csv(transactions_file, index=False, quoting=csv.QUOTE_ALL)
    
    print(f"Successfully fixed CSV format issues in {transactions_file}")
    print(f"All fields are now properly quoted to handle commas in merchant names")
    
    # Verify the file can be read by pandas
    try:
        df_verify = pd.read_csv(transactions_file)
        print(f"Verification successful: File can be read correctly with {len(df_verify)} rows")
    except Exception as e:
        print(f"Verification failed: {str(e)}")

if __name__ == "__main__":
    main()
