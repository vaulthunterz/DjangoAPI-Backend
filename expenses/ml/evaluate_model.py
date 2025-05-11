"""
Main script to evaluate the expense categorization model
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from scripts
from scripts.evaluate import evaluate_model

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Evaluate expense categorization model')
    parser.add_argument('--data', type=str, help='Path to evaluation data CSV file')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory with trained models')
    parser.add_argument('--output-dir', type=str, default='evaluation', help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate the model
    results = evaluate_model(
        data_path=args.data,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    return results

if __name__ == "__main__":
    main()
