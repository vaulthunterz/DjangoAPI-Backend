"""
Training script for expense categorization models
"""

import os
import sys
import logging
import argparse
import pandas as pd
import time
from tabulate import tabulate

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
    from expenses.ml.utils.model_utils import save_performance_metrics, print_performance_summary
    from expenses.ml.utils.data_utils import load_training_data
except ImportError:
    # Try importing from current directory structure
    sys.path.append(os.path.dirname(current_dir))
    from core.model import ExpenseCategorizer
    from utils.model_utils import save_performance_metrics, print_performance_summary
    from utils.data_utils import load_training_data

def train_model(data_path=None, models_dir='models', sample_size=None, test_size=0.2, random_state=42):
    """Train the expense categorization model"""
    start_time = time.time()
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    if data_path is None:
        # Use default path
        ml_dir = os.path.dirname(current_dir)
        data_path = os.path.join(ml_dir, 'training_data', 'transactions.csv')
    
    logger.info(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} transactions with {df['category'].nunique()} categories")
    
    # Take a sample if requested
    if sample_size is not None and sample_size < len(df):
        logger.info(f"Taking a sample of {sample_size} transactions...")
        df = df.sample(sample_size, random_state=random_state)
        logger.info(f"Sample contains {df['category'].nunique()} categories")
    
    # Initialize the model
    logger.info("Initializing expense categorizer...")
    categorizer = ExpenseCategorizer(models_dir=models_dir)
    
    # Train the model
    logger.info(f"Training model with data from {data_path}...")
    results = categorizer.train(data_path=data_path, test_size=test_size, random_state=random_state)
    
    # Save performance metrics
    logger.info("Saving performance metrics...")
    save_performance_metrics(results, models_dir=models_dir)
    
    # Print performance summary
    print_performance_summary(results)
    
    # Calculate total time
    total_time = time.time() - start_time
    logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return categorizer, results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train expense categorization model')
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--sample', type=int, help='Number of transactions to sample')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0.0-1.0)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Train the model
    categorizer, results = train_model(
        data_path=args.data,
        models_dir=args.models_dir,
        sample_size=args.sample,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    return results

if __name__ == "__main__":
    main()
