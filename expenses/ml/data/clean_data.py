"""
Data cleaning and preparation utilities
"""

import os
import sys
import pandas as pd
import csv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_csv_file(input_file, output_file=None):
    """Clean a CSV file by fixing common issues"""
    if output_file is None:
        # Use the same file for input and output
        output_file = input_file

    # Create a backup of the original file
    backup_file = f"{input_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Creating backup of original file: {backup_file}")

    try:
        # Read the file with pandas
        df = pd.read_csv(input_file, encoding='utf-8')
        logger.info(f"Read {len(df)} rows from {input_file}")

        # Clean up column names
        df.columns = [col.strip().lower() for col in df.columns]

        # Ensure required columns exist
        required_columns = ['description', 'category', 'subcategory']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in the CSV file")
                return False

        # Fill missing values
        if 'merchant' in df.columns:
            df['merchant'] = df['merchant'].fillna('')
        else:
            df['merchant'] = ''

        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        else:
            df['amount'] = 0

        if 'is_expense' in df.columns:
            df['is_expense'] = pd.to_numeric(df['is_expense'], errors='coerce').fillna(1).astype(int)
        else:
            df['is_expense'] = 1

        # Remove rows with missing required values
        initial_count = len(df)
        df = df.dropna(subset=['description', 'category', 'subcategory'])
        dropped_count = initial_count - len(df)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with missing required values")

        # Convert to string and strip whitespace
        for col in ['description', 'merchant', 'category', 'subcategory']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        # Convert amount to float
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)

        # Convert is_expense to int
        if 'is_expense' in df.columns:
            df['is_expense'] = df['is_expense'].astype(int)

        # Save the backup
        with open(backup_file, 'w', newline='', encoding='utf-8') as f:
            with open(input_file, 'r', encoding='utf-8') as original:
                f.write(original.read())

        # Save the cleaned data
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Saved {len(df)} rows to {output_file}")

        return True
    except Exception as e:
        logger.error(f"Error cleaning CSV file: {str(e)}")
        return False

def organize_training_data():
    """Organize the training_data directory"""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.dirname(current_dir)
    training_data_dir = os.path.join(ml_dir, 'training_data')

    # Create a scripts directory for utility scripts
    scripts_dir = os.path.join(training_data_dir, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)

    # Create a samples directory for sample data
    samples_dir = os.path.join(training_data_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Move Python scripts to the scripts directory
    for item in os.listdir(training_data_dir):
        item_path = os.path.join(training_data_dir, item)

        # Move Python scripts
        if item.endswith('.py'):
            target_path = os.path.join(scripts_dir, item)
            try:
                os.rename(item_path, target_path)
                logger.info(f"Moved {item} to scripts directory")
            except Exception as e:
                logger.error(f"Error moving {item}: {str(e)}")

        # Move sample CSV files
        elif item.startswith('sample_') and item.endswith('.csv'):
            target_path = os.path.join(samples_dir, item)
            try:
                os.rename(item_path, target_path)
                logger.info(f"Moved {item} to samples directory")
            except Exception as e:
                logger.error(f"Error moving {item}: {str(e)}")

    # Clean the main transactions.csv file
    transactions_file = os.path.join(training_data_dir, 'transactions.csv')
    if os.path.exists(transactions_file):
        logger.info("Cleaning transactions.csv file...")
        clean_csv_file(transactions_file)

    logger.info("Training data directory organized successfully")

if __name__ == "__main__":
    organize_training_data()
