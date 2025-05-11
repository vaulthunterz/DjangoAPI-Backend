"""
Script to fix the transactions.csv file with proper handling of quotes and commas
"""

import os
import csv
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

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    transactions_file = os.path.join(base_dir, 'transactions.csv')
    temp_file = os.path.join(base_dir, 'transactions_temp.csv')
    
    # Backup the original file
    backup_file(transactions_file)
    
    print("Reading transactions file...")
    
    # Read the file as text
    with open(transactions_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # Split into lines
    lines = content.strip().split('\n')
    
    # Process header
    header = lines[0].strip()
    # Remove all quotes from header
    header = header.replace('"', '')
    
    # Process data rows
    processed_lines = [header]
    
    for i, line in enumerate(lines[1:], start=1):
        # Skip empty lines
        if not line.strip():
            continue
        
        # Remove all quotes
        clean_line = line.replace('"', '')
        
        # Split by commas
        parts = clean_line.split(',')
        
        # Check if we have the expected number of columns (5)
        if len(parts) == 5:
            # Line is good, add it as is
            processed_lines.append(clean_line)
        elif len(parts) > 5:
            # Too many commas - likely a comma in the merchant name
            # Reconstruct the line with proper quoting for merchant
            description = parts[0]
            category = parts[1]
            subcategory = parts[2]
            
            # Merchant might have commas, so we need to find where it ends
            # We know amount is the last field and should be numeric
            # Start from the end and work backwards until we find a numeric field
            amount = parts[-1]
            merchant_parts = parts[3:-1]  # All parts between subcategory and amount
            merchant = ','.join(merchant_parts)
            
            # Create a new line with proper quoting for merchant
            new_line = f'{description},{category},{subcategory},"{merchant}",{amount}'
            processed_lines.append(new_line)
            
            if i <= 10:  # Print first few fixed lines for debugging
                print(f"Fixed line {i}: {line} -> {new_line}")
        else:
            # Too few columns - skip this line
            print(f"Skipping line {i} with too few columns: {line}")
    
    # Write the processed lines to the file
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        outfile.write('\n'.join(processed_lines))
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully fixed CSV format issues in {transactions_file}")
    print(f"Processed {len(processed_lines)-1} data rows")
    
    # Verify the file can be read by csv module
    try:
        with open(transactions_file, 'r', encoding='utf-8', newline='') as verify_file:
            reader = csv.reader(verify_file)
            row_count = sum(1 for _ in reader) - 1  # Subtract 1 for header
        print(f"Verification successful: File can be read correctly with {row_count} rows")
    except Exception as e:
        print(f"Verification failed: {str(e)}")

if __name__ == "__main__":
    main()
