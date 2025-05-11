"""
Script to fix CSV format issues in transactions.csv by:
1. Removing any newlines within fields
2. Ensuring all fields are properly quoted
3. Fixing any other CSV formatting issues
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
    
    # Read the entire file as text
    with open(transactions_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Process header
    header = lines[0].split(',')
    expected_columns = len(header)
    print(f"Header has {expected_columns} columns: {header}")
    
    # Process data rows
    processed_rows = []
    processed_rows.append(header)  # Add header
    
    for i, line in enumerate(lines[1:], start=1):
        if not line.strip():  # Skip empty lines
            continue
            
        # Split by commas, but be careful about quoted fields
        fields = []
        current_field = ""
        in_quotes = False
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            
            # Add character to current field
            current_field += char
            
            # If we're at a comma and not inside quotes, end the field
            if char == ',' and not in_quotes:
                # Remove the trailing comma
                current_field = current_field[:-1]
                fields.append(current_field)
                current_field = ""
        
        # Add the last field
        if current_field:
            fields.append(current_field)
        
        # Check if we have the expected number of columns
        if len(fields) != expected_columns:
            print(f"Line {i+1} has {len(fields)} fields instead of {expected_columns}: {line}")
            
            # Try to fix the line
            if len(fields) > expected_columns:
                # Combine extra fields into the description
                description = ','.join(fields[:-4])
                category = fields[-4]
                subcategory = fields[-3]
                merchant = fields[-2]
                amount = fields[-1]
                fields = [description, category, subcategory, merchant, amount]
            elif len(fields) < expected_columns:
                # This shouldn't happen with our data, but just in case
                # Pad with empty fields
                fields.extend([''] * (expected_columns - len(fields)))
        
        # Clean each field
        for j in range(len(fields)):
            # Remove any newlines or carriage returns
            fields[j] = fields[j].replace('\n', ' ').replace('\r', ' ')
            
            # Remove any existing quotes
            fields[j] = fields[j].replace('"', '')
            
            # Trim whitespace
            fields[j] = fields[j].strip()
        
        processed_rows.append(fields)
    
    print(f"Processed {len(processed_rows)-1} data rows")
    
    # Write the processed data to a new CSV file
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        writer.writerows(processed_rows)
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully fixed CSV format issues in {transactions_file}")
    
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
