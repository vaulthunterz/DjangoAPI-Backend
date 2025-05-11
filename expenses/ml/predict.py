"""
Main script to make predictions with the expense categorization model
"""

import os
import sys
import logging
import argparse
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from scripts
from scripts.predict import predict_transaction, predict_from_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict expense categories')
    parser.add_argument('--file', type=str, help='Path to CSV file with transactions')
    parser.add_argument('--description', type=str, help='Transaction description')
    parser.add_argument('--merchant', type=str, default='', help='Merchant name')
    parser.add_argument('--amount', type=float, default=0, help='Transaction amount')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory with trained models')
    parser.add_argument('--output', type=str, help='Output file for predictions (CSV)')
    
    args = parser.parse_args()
    
    # Check if file or description is provided
    if args.file:
        # Predict from file
        predictions = predict_from_file(
            file_path=args.file,
            models_dir=args.models_dir,
            output_file=args.output
        )
    elif args.description:
        # Predict single transaction
        transaction = {
            'description': args.description,
            'merchant': args.merchant,
            'amount': args.amount
        }
        prediction = predict_transaction(transaction, models_dir=args.models_dir)
        
        # Print prediction
        print(json.dumps(prediction, indent=2))
    else:
        parser.error("Either --file or --description must be provided")

if __name__ == "__main__":
    main()
