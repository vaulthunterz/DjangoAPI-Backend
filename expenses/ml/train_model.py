import os
import sys
import logging
from pprint import pformat
from tabulate import tabulate

# Add the parent directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from expenses.ml.utils import load_training_data
    from expenses.ml.model_training import ExpenseCategoryClassifier
except ImportError:
    from utils import load_training_data
    from model_training import ExpenseCategoryClassifier

def train_model():
    """
    Train the Random Forest classifier with the provided transaction data.
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("Loading training data...")
        training_data = load_training_data()
        
        logger.info("Initializing classifier...")
        classifier = ExpenseCategoryClassifier()
        
        logger.info("Training model...")
        results = classifier.train_model(training_data)
        
        # Format performance metrics in tables
        logger.info("\n=== Model Performance Summary ===")
        
        # Overall category accuracy
        logger.info(f"\nCategory Model Accuracy: {results['category_accuracy']:.2%}")
        
        # Subcategory accuracies table
        subcategory_data = [[category, f"{accuracy:.2%}"] 
                           for category, accuracy in sorted(results['subcategory_accuracies'].items())]
        logger.info("\nSubcategory Model Performance by Category:")
        logger.info("\n" + tabulate(subcategory_data, 
                                  headers=['Category', 'Accuracy'],
                                  tablefmt='grid'))
        
        # Feature importance table
        feature_data = [[feature, f"{importance:.4f}"] 
                       for feature, importance in sorted(
                           results['category_feature_importance'].items(),
                           key=lambda x: x[1],
                           reverse=True
                       )[:10]]
        logger.info("\nTop 10 Important Features:")
        logger.info("\n" + tabulate(feature_data,
                                  headers=['Feature', 'Importance Score'],
                                  tablefmt='grid'))
        
        # Category-Subcategory mapping table
        mapping_data = []
        for category, subcategories in sorted(results['category_subcategory_mapping'].items()):
            mapping_data.append([
                category,
                len(subcategories),
                ", ".join(sorted(subcategories)[:3]) + 
                ("..." if len(subcategories) > 3 else "")
            ])
        
        logger.info("\nCategory to Subcategory Overview:")
        logger.info("\n" + tabulate(mapping_data,
                                  headers=['Category', '# Subcategories', 'Sample Subcategories'],
                                  tablefmt='grid'))
        
        # Performance statistics
        stats = {
            'Total Categories': len(results['unique_categories']),
            'Avg Subcategory Accuracy': sum(results['subcategory_accuracies'].values()) / 
                                      len(results['subcategory_accuracies']),
            'Best Performing Category': max(results['subcategory_accuracies'].items(), 
                                          key=lambda x: x[1])[0],
            'Worst Performing Category': min(results['subcategory_accuracies'].items(), 
                                           key=lambda x: x[1])[0]
        }
        
        stats_data = [[k, v if isinstance(v, str) else f"{v:.2%}"] 
                     for k, v in stats.items()]
        logger.info("\nOverall Statistics:")
        logger.info("\n" + tabulate(stats_data,
                                  headers=['Metric', 'Value'],
                                  tablefmt='grid'))
        
        return True
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return False

if __name__ == "__main__":
    train_model() 