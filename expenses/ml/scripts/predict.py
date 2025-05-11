"""
Prediction script for expense categorization models
"""

import os
import sys
import logging
import argparse
import json
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from relative paths
try:
    from expenses.ml.core.model import ExpenseCategorizer
except ImportError:
    # Try importing from current directory structure
    sys.path.append(os.path.dirname(current_dir))
    from core.model import ExpenseCategorizer

def predict_transaction(transaction, models_dir='models'):
    """Predict category and subcategory for a transaction"""
    # Initialize the model
    logger.info("Loading trained model...")
    categorizer = ExpenseCategorizer(models_dir=models_dir)
    categorizer.load_models()
    
    # Make prediction
    logger.info("Making prediction...")
    prediction = categorizer.predict(transaction)
    
    return prediction

def predict_from_file(file_path, models_dir='models', output_file=None):
    """Predict categories for transactions in a CSV file"""
    # Initialize the model
    logger.info("Loading trained model...")
    categorizer = ExpenseCategorizer(models_dir=models_dir)
    categorizer.load_models()
    
    # Load data
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = []
    for _, row in df.iterrows():
        transaction = {
            'description': row['description'],
            'merchant': row.get('merchant', '') if 'merchant' in row else '',
            'amount': row.get('amount', 0) if 'amount' in row else 0
        }
        prediction = categorizer.predict(transaction)
        
        # Add to results
        result = {
            'description': transaction['description'],
            'merchant': transaction['merchant'],
            'amount': transaction['amount'],
            'predicted_category': prediction['category'],
            'predicted_subcategory': prediction['subcategory']
        }
        
        # Add true values if available
        if 'category' in row:
            result['true_category'] = row['category']
        if 'subcategory' in row:
            result['true_subcategory'] = row['subcategory']
        
        predictions.append(result)
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Calculate accuracy if true values are available
    if 'true_category' in pred_df.columns:
        category_accuracy = (pred_df['predicted_category'] == pred_df['true_category']).mean()
        logger.info(f"Category accuracy: {category_accuracy*100:.2f}%")
    
    if 'true_subcategory' in pred_df.columns and 'true_category' in pred_df.columns:
        # Only consider rows where category prediction is correct
        correct_category_mask = pred_df['true_category'] == pred_df['predicted_category']
        subcategory_accuracy = (
            pred_df.loc[correct_category_mask, 'predicted_subcategory'] == 
            pred_df.loc[correct_category_mask, 'true_subcategory']
        ).mean()
        logger.info(f"Subcategory accuracy (when category is correct): {subcategory_accuracy*100:.2f}%")
    
    # Save predictions if output file is specified
    if output_file:
        pred_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
    
    return pred_df

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
