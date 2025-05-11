"""
Script to consolidate all synthetic transaction data into a single transactions.csv file.

This script:
1. Finds all CSV files in the synthetic data directories
2. Reads each CSV file and validates its structure
3. Combines all valid data into a single DataFrame
4. Saves the combined data to a new transactions.csv file (after backing up the original)
"""

import os
import pandas as pd
import shutil
from datetime import datetime

# Define the expected columns for validation
EXPECTED_COLUMNS = ['description', 'category', 'subcategory', 'merchant', 'amount']

def find_csv_files(base_dir):
    """Find all CSV files in the given directory and its subdirectories."""
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv') and 'transactions' in file.lower():
                csv_files.append(os.path.join(root, file))
    return csv_files

def validate_csv_structure(file_path):
    """Validate that the CSV file has the expected structure."""
    try:
        df = pd.read_csv(file_path)
        # Check if all expected columns are present
        if all(col in df.columns for col in EXPECTED_COLUMNS):
            return df
        else:
            print(f"Warning: File {file_path} is missing required columns. Skipping.")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None

def backup_original_file(file_path):
    """Create a backup of the original transactions.csv file."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of original file at {backup_path}")
        return True
    return False

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    synthetic_data_dir = os.path.join(base_dir, 'Synthetic data')
    output_file = os.path.join(base_dir, 'transactions.csv')
    
    # Backup the original file
    backup_original_file(output_file)
    
    # Find all CSV files
    print("Finding CSV files...")
    csv_files = find_csv_files(synthetic_data_dir)
    print(f"Found {len(csv_files)} CSV files.")
    
    # Also include the original transactions.csv if it exists
    if os.path.exists(output_file):
        csv_files.append(output_file)
        print("Added original transactions.csv to the list.")
    
    # Read and validate each file
    print("Reading and validating files...")
    dataframes = []
    for file_path in csv_files:
        df = validate_csv_structure(file_path)
        if df is not None:
            print(f"Successfully read {file_path} with {len(df)} rows.")
            dataframes.append(df)
    
    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        duplicate_count = original_count - len(combined_df)
        print(f"Removed {duplicate_count} duplicate rows.")
        
        # Save the combined data
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(combined_df)} transactions to {output_file}")
    else:
        print("No valid data found. No output file created.")

if __name__ == "__main__":
    main()
