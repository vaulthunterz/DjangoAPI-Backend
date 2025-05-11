"""
Script to ensure all description fields in transactions.csv are properly quoted,
especially those containing commas
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
    
    # Read the file using pandas to handle CSV parsing correctly
    try:
        df = pd.read_csv(transactions_file)
        print(f"Successfully read {len(df)} rows from {transactions_file}")
        
        # Check for descriptions with commas
        descriptions_with_commas = df['description'].str.contains(',', na=False)
        num_with_commas = descriptions_with_commas.sum()
        print(f"Found {num_with_commas} descriptions containing commas")
        
        # Write the file back with proper quoting
        df.to_csv(temp_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Wrote data to {temp_file} with all fields properly quoted")
        
        # Replace the original file with the temporary file
        os.remove(transactions_file)
        os.rename(temp_file, transactions_file)
        print(f"Replaced {transactions_file} with properly quoted version")
        
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        
        # If pandas approach fails, try a more manual approach
        print("Trying manual CSV processing...")
        
        # Open the output file
        with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
            # Write the header
            outfile.write("description,category,subcategory,merchant,amount\n")
            
            # Process each line
            fixed_count = 0
            with open(transactions_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                
                for i, row in enumerate(reader, start=1):
                    if len(row) >= 5:  # Ensure we have at least 5 fields
                        description = row[0].strip()
                        category = row[1].strip()
                        subcategory = row[2].strip()
                        merchant = row[3].strip()
                        amount = row[4].strip()
                        
                        # Always quote the description field
                        if not description.startswith('"') or not description.endswith('"'):
                            description = f'"{description}"'
                        
                        # Quote other fields if they contain commas
                        if ',' in category and (not category.startswith('"') or not category.endswith('"')):
                            category = f'"{category}"'
                        if ',' in subcategory and (not subcategory.startswith('"') or not subcategory.endswith('"')):
                            subcategory = f'"{subcategory}"'
                        if ',' in merchant and (not merchant.startswith('"') or not merchant.endswith('"')):
                            merchant = f'"{merchant}"'
                        
                        # Write the fixed line
                        fixed_line = f"{description},{category},{subcategory},{merchant},{amount}\n"
                        outfile.write(fixed_line)
                        fixed_count += 1
                        
                        if i <= 5 or i % 1000 == 0:  # Print progress
                            print(f"Processed line {i}: {fixed_line.strip()}")
            
            # Replace the original file with the temporary file
            os.remove(transactions_file)
            os.rename(temp_file, transactions_file)
            print(f"Successfully fixed CSV format using manual approach")
            print(f"Fixed {fixed_count} lines")
    
    # Verify the file can be read correctly
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
