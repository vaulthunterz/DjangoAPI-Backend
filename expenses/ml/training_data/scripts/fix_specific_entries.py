"""
Script to fix specific entries in transactions.csv that have been incorrectly parsed
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
    
    # Open the output file
    with open(temp_file, 'w', encoding='utf-8', newline='') as outfile:
        # Write the header
        outfile.write("description,category,subcategory,merchant,amount\n")
        
        # Read the input file line by line
        fixed_count = 0
        skipped_count = 0
        special_fixes = 0
        
        with open(transactions_file, 'r', encoding='utf-8') as infile:
            # Skip header
            next(infile)
            
            for i, line in enumerate(infile, start=1):
                line = line.strip()
                
                # Special case for Paracetamol, Vicks, Strepsils
                if '"Paracetamol","Vicks","Strepsils"' in line:
                    fixed_line = '"Paracetamol, Vicks, Strepsils","Health & Fitness","Over-the-Counter Medicines","Avenue Hospital",72.71\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed special case line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for Strepsils
                elif '"Strepsils"' in line or line.startswith('Strepsils,'):
                    fixed_line = '"Strepsils","Health & Fitness","Over-the-Counter Medicines","Pharmacy",100.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed Strepsils line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for cough syrup
                elif '"cough syrup"' in line or line.startswith('cough syrup,'):
                    fixed_line = '"cough syrup","Health & Fitness","Over-the-Counter Medicines","Pharmacy",150.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed cough syrup line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for floss
                elif '"floss"' in line or line.startswith('floss,'):
                    fixed_line = '"floss","Personal Care & Grooming","Dental Care","Supermarket",80.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed floss line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for pedicure
                elif '"pedicure"' in line or line.startswith('pedicure,'):
                    fixed_line = '"pedicure","Personal Care & Grooming","Manicure & Pedicure","Beauty Salon",500.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed pedicure line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for pens
                elif '"pens"' in line or line.startswith('pens,'):
                    fixed_line = '"pens","Shopping","Books & Stationery","Bookstore",120.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed pens line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for premium
                elif '"premium"' in line or line.startswith('premium,'):
                    fixed_line = '"premium","Utilities","Electricity","KPLC",1000.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed premium line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for sabuni
                elif '"sabuni"' in line or line.startswith('sabuni,'):
                    fixed_line = '"sabuni","Shopping","Household Supplies","Supermarket",150.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed sabuni line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # Special case for tomatoes
                elif '"tomatoes"' in line or line.startswith('tomatoes,'):
                    fixed_line = '"tomatoes","Food & Dining","Groceries","Market",80.00\n'
                    outfile.write(fixed_line)
                    special_fixes += 1
                    print(f"Fixed tomatoes line {i}: {line} -> {fixed_line.strip()}")
                    continue
                
                # For all other lines, try to parse and fix if needed
                else:
                    try:
                        # Try to parse the line using CSV reader
                        reader = csv.reader([line])
                        fields = next(reader)
                        
                        if len(fields) >= 5:
                            # Extract fields
                            description = fields[0]
                            category = fields[1]
                            subcategory = fields[2]
                            merchant = fields[3]
                            amount = fields[4]
                            
                            # Write the fixed line
                            fixed_line = f'"{description}","{category}","{subcategory}","{merchant}",{amount}\n'
                            outfile.write(fixed_line)
                            fixed_count += 1
                        else:
                            # Skip lines with too few fields
                            print(f"Skipping line {i} with too few fields: {line}")
                            skipped_count += 1
                    except Exception as e:
                        print(f"Error parsing line {i}: {e}")
                        print(f"Line content: {line}")
                        skipped_count += 1
    
    # Replace the original file with the temporary file
    os.remove(transactions_file)
    os.rename(temp_file, transactions_file)
    
    print(f"Successfully fixed CSV format issues in {transactions_file}")
    print(f"Fixed {fixed_count} regular lines, {special_fixes} special cases, skipped {skipped_count} lines")
    
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
