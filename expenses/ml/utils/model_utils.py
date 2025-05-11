"""
Model utility functions for expense categorization models
"""

import os
import json
import logging
import numpy as np
from tabulate import tabulate
import joblib

logger = logging.getLogger(__name__)

def save_performance_metrics(results, models_dir='models'):
    """Save performance metrics to a file"""
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Clean up old performance files
    performance_pkl_path = os.path.join(models_dir, 'performance.pkl')
    performance_path = os.path.join(models_dir, 'performance.joblib')
    json_path = os.path.join(models_dir, 'performance.json')

    try:
        # Remove old performance files if they exist
        for file_path in [performance_pkl_path, performance_path, json_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed old performance file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up old performance files: {str(e)}")

    # Save performance metrics using joblib
    joblib.dump(results, performance_path)

    # Also save as JSON for easier viewing
    # Convert numpy values to Python types for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == 'subcategory_accuracies':
            json_results[key] = {k: float(v) for k, v in value.items()}
        elif key == 'individual_model_metrics':
            # This is already a list of dictionaries with Python types
            json_results[key] = value
        elif isinstance(value, (int, float, str, bool, list, dict)):
            json_results[key] = value
        else:
            try:
                json_results[key] = float(value)
            except (TypeError, ValueError):
                json_results[key] = str(value)

    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"Performance metrics saved to {performance_path} and {json_path}")

    return performance_path

def print_performance_summary(results):
    """Print a summary of model performance"""
    logger.info("\n=== Model Performance Summary ===")

    # Print category model metrics
    category_accuracy = results.get('category_accuracy', 0)
    category_precision = results.get('category_precision', 0)
    category_recall = results.get('category_recall', 0)
    category_f1 = results.get('category_f1', 0)

    logger.info("\nCategory Model Performance:")
    metrics_table = [
        ["Accuracy", f"{category_accuracy*100:.2f}%"],
        ["Precision", f"{category_precision*100:.2f}%"],
        ["Recall", f"{category_recall*100:.2f}%"],
        ["F1 Score", f"{category_f1*100:.2f}%"]
    ]
    logger.info("\n" + tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

    # Print individual model metrics if available
    individual_model_metrics = results.get('individual_model_metrics', [])
    if individual_model_metrics:
        logger.info("\nIndividual Model Performance:")
        model_table = []
        for i, model_metrics in enumerate(individual_model_metrics):
            model_table.append([
                i+1,
                model_metrics.get('model_name', 'Unknown'),
                f"{model_metrics.get('accuracy', 0)*100:.2f}%",
                f"{model_metrics.get('precision', 0)*100:.2f}%",
                f"{model_metrics.get('recall', 0)*100:.2f}%",
                f"{model_metrics.get('f1_score', 0)*100:.2f}%"
            ])
        logger.info("\n" + tabulate(model_table,
                                   headers=["#", "Model", "Accuracy", "Precision", "Recall", "F1 Score"],
                                   tablefmt="grid"))

    # Print subcategory accuracies
    subcategory_accuracies = results.get('subcategory_accuracies', {})
    if subcategory_accuracies:
        # Calculate average subcategory accuracy
        avg_subcategory_accuracy = np.mean(list(subcategory_accuracies.values()))
        logger.info(f"\nAverage subcategory accuracy: {avg_subcategory_accuracy*100:.2f}%")

        # Print top 5 best performing subcategory models
        logger.info("\nTop 5 best performing subcategory models:")
        top_subcategories = sorted(subcategory_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
        table_data = [(i+1, category, f"{accuracy*100:.2f}%") for i, (category, accuracy) in enumerate(top_subcategories)]
        logger.info("\n" + tabulate(table_data, headers=["#", "Category", "Accuracy"], tablefmt="grid"))

        # Print bottom 5 worst performing subcategory models
        logger.info("\nBottom 5 worst performing subcategory models:")
        bottom_subcategories = sorted(subcategory_accuracies.items(), key=lambda x: x[1])[:5]
        table_data = [(i+1, category, f"{accuracy*100:.2f}%") for i, (category, accuracy) in enumerate(bottom_subcategories)]
        logger.info("\n" + tabulate(table_data, headers=["#", "Category", "Accuracy"], tablefmt="grid"))

    # Print training time
    training_time = results.get('training_time', 0)
    logger.info(f"\nTraining time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # Print feature importance if available
    feature_importance = results.get('category_feature_importance', {})
    if feature_importance:
        logger.info("\nTop 10 important features:")
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        table_data = [(i+1, feature, f"{importance:.4f}") for i, (feature, importance) in enumerate(top_features)]
        logger.info("\n" + tabulate(table_data, headers=["#", "Feature", "Importance"], tablefmt="grid"))
