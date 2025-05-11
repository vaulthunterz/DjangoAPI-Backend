"""
Evaluation script for expense categorization models
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
    from expenses.ml.utils.data_utils import load_training_data
except ImportError:
    # Try importing from current directory structure
    sys.path.append(os.path.dirname(current_dir))
    from core.model import ExpenseCategorizer
    from utils.data_utils import load_training_data

def evaluate_model(data_path=None, models_dir='models', output_dir='evaluation', test_size=0.2, random_state=42):
    """Evaluate the expense categorization model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

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

    # Initialize the model
    logger.info("Loading trained model...")
    categorizer = ExpenseCategorizer(models_dir=models_dir)
    categorizer.load_models()

    # Evaluate on test data
    logger.info("Evaluating model on test data...")

    # Make predictions
    predictions = []
    for _, row in df.iterrows():
        transaction = {
            'description': row['description'],
            'merchant': row.get('merchant', ''),
            'amount': row.get('amount', 0)
        }
        prediction = categorizer.predict(transaction)
        predictions.append({
            'category_true': row['category'],
            'subcategory_true': row['subcategory'],
            'category_pred': prediction['category'],
            'subcategory_pred': prediction['subcategory']
        })

    # Convert to DataFrame
    pred_df = pd.DataFrame(predictions)

    # Calculate category accuracy
    category_accuracy = accuracy_score(pred_df['category_true'], pred_df['category_pred'])
    logger.info(f"Category accuracy: {category_accuracy*100:.2f}%")

    # Calculate subcategory accuracy
    # Only consider rows where category prediction is correct
    correct_category_mask = pred_df['category_true'] == pred_df['category_pred']
    subcategory_accuracy = accuracy_score(
        pred_df.loc[correct_category_mask, 'subcategory_true'],
        pred_df.loc[correct_category_mask, 'subcategory_pred']
    )
    logger.info(f"Subcategory accuracy (when category is correct): {subcategory_accuracy*100:.2f}%")

    # Generate classification report
    logger.info("\nCategory Classification Report:")
    category_report = classification_report(
        pred_df['category_true'],
        pred_df['category_pred'],
        output_dict=True
    )

    # Convert to DataFrame for easier viewing
    category_report_df = pd.DataFrame(category_report).transpose()
    logger.info("\n" + tabulate(category_report_df, headers='keys', tablefmt='grid'))

    # Save classification report
    category_report_df.to_csv(os.path.join(output_dir, 'category_report.csv'))

    # Generate confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(pred_df['category_true'], pred_df['category_pred'])

    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    sns.heatmap(cm_norm, annot=False, cmap='Blues', fmt='.2f')
    plt.title('Category Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    # Calculate per-category subcategory accuracy
    subcategory_accuracies = {}
    for category in df['category'].unique():
        category_mask = (pred_df['category_true'] == category) & (pred_df['category_pred'] == category)
        if category_mask.sum() > 0:
            sub_accuracy = accuracy_score(
                pred_df.loc[category_mask, 'subcategory_true'],
                pred_df.loc[category_mask, 'subcategory_pred']
            )
            subcategory_accuracies[category] = sub_accuracy

    # Print subcategory accuracies
    logger.info("\nSubcategory Accuracies by Category:")
    subcategory_data = [(category, f"{accuracy*100:.2f}%") for category, accuracy in
                        sorted(subcategory_accuracies.items(), key=lambda x: x[1], reverse=True)]
    logger.info("\n" + tabulate(subcategory_data, headers=['Category', 'Subcategory Accuracy'], tablefmt='grid'))

    # Save subcategory accuracies
    pd.DataFrame(subcategory_data, columns=['Category', 'Subcategory Accuracy']).to_csv(
        os.path.join(output_dir, 'subcategory_accuracies.csv'), index=False
    )

    logger.info(f"Evaluation results saved to {output_dir}")

    return {
        'category_accuracy': category_accuracy,
        'subcategory_accuracy': subcategory_accuracy,
        'subcategory_accuracies': subcategory_accuracies
    }

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
