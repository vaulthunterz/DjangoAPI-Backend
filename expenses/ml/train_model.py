"""
Main script to train the expense categorization model
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from scripts
from scripts.train import train_model

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
