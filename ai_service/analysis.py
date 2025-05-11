"""
Model Analysis Module

This module provides functionality for analyzing the performance of ML models,
particularly focusing on identifying misclassified categories.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class ModelAnalyzer:
    """
    Analyzes model performance, particularly focusing on misclassifications.
    """
    
    def __init__(self, prediction_service=None):
        """
        Initialize the model analyzer.
        
        Args:
            prediction_service: Service used to make predictions
        """
        self.prediction_service = prediction_service
        
    def analyze_misclassifications(self, 
                                  transactions: List[Dict[str, Any]], 
                                  model_type: str = 'custom') -> Dict[str, Any]:
        """
        Analyze misclassifications in a set of transactions.
        
        Args:
            transactions: List of transaction dictionaries with 'description', 'merchant',
                         'category' (actual category), and optionally 'subcategory'
            model_type: Type of model to use ('custom' or 'default')
            
        Returns:
            Dict containing analysis results
        """
        if not transactions:
            return {"error": "No transactions provided for analysis"}
            
        # Validate transactions have required fields
        for i, transaction in enumerate(transactions):
            if 'description' not in transaction:
                return {"error": f"Transaction at index {i} missing 'description' field"}
            if 'category' not in transaction:
                return {"error": f"Transaction at index {i} missing 'category' field"}
        
        # Make predictions
        predictions = []
        for transaction in transactions:
            try:
                # Prepare input data
                input_data = {
                    'description': transaction['description'],
                    'merchant': transaction.get('merchant', '')
                }
                
                # Get prediction based on model type
                if model_type == 'custom' and self.prediction_service:
                    prediction = self.prediction_service.predict_with_custom_model(input_data)
                    predicted_category = prediction.get('category', '')
                    predicted_subcategory = prediction.get('subcategory', '')
                    confidence = prediction.get('confidence', 0)
                elif self.prediction_service:
                    prediction = self.prediction_service.predict(input_data)
                    predicted_category = prediction.get('category', '')
                    predicted_subcategory = prediction.get('subcategory', '')
                    confidence = prediction.get('category_confidence', 0)
                else:
                    return {"error": "Prediction service not available"}
                
                # Store prediction result
                predictions.append({
                    'description': transaction['description'],
                    'merchant': transaction.get('merchant', ''),
                    'actual_category': transaction['category'],
                    'actual_subcategory': transaction.get('subcategory', ''),
                    'predicted_category': predicted_category,
                    'predicted_subcategory': predicted_subcategory,
                    'confidence': confidence,
                    'is_correct': predicted_category == transaction['category']
                })
                
            except Exception as e:
                logger.error(f"Error predicting transaction: {str(e)}")
                # Continue with next transaction
        
        # Analyze results
        return self._analyze_results(predictions)
    
    def _analyze_results(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze prediction results to identify patterns in misclassifications.
        
        Args:
            predictions: List of prediction results
            
        Returns:
            Dict containing analysis results
        """
        if not predictions:
            return {"error": "No predictions to analyze"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(predictions)
        
        # Overall accuracy
        total_count = len(df)
        correct_count = df['is_correct'].sum()
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        # Confusion matrix (actual vs predicted categories)
        confusion = defaultdict(Counter)
        for _, row in df.iterrows():
            actual = row['actual_category']
            predicted = row['predicted_category']
            confusion[actual][predicted] += 1
        
        # Convert confusion matrix to list format for easier JSON serialization
        confusion_matrix = []
        for actual, predictions_counter in confusion.items():
            for predicted, count in predictions_counter.items():
                confusion_matrix.append({
                    'actual': actual,
                    'predicted': predicted,
                    'count': count
                })
        
        # Most frequently misclassified categories
        misclassified = df[~df['is_correct']]
        misclassified_counts = misclassified['actual_category'].value_counts().to_dict()
        
        # Categories with highest error rates
        category_stats = []
        for category in df['actual_category'].unique():
            category_df = df[df['actual_category'] == category]
            total = len(category_df)
            correct = category_df['is_correct'].sum()
            error_rate = 1 - (correct / total) if total > 0 else 0
            
            category_stats.append({
                'category': category,
                'total_samples': total,
                'correct_predictions': correct,
                'error_rate': error_rate
            })
        
        # Sort by error rate (highest first)
        category_stats.sort(key=lambda x: x['error_rate'], reverse=True)
        
        # Common misclassification patterns
        misclassification_patterns = []
        for actual in df['actual_category'].unique():
            actual_df = df[(df['actual_category'] == actual) & (~df['is_correct'])]
            if len(actual_df) > 0:
                common_wrong_predictions = actual_df['predicted_category'].value_counts().to_dict()
                
                for predicted, count in common_wrong_predictions.items():
                    misclassification_patterns.append({
                        'actual': actual,
                        'predicted': predicted,
                        'count': count,
                        'examples': actual_df[actual_df['predicted_category'] == predicted][['description', 'merchant']].head(3).to_dict('records')
                    })
        
        # Sort by count (highest first)
        misclassification_patterns.sort(key=lambda x: x['count'], reverse=True)
        
        # Return analysis results
        return {
            'total_transactions': total_count,
            'correct_predictions': correct_count,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix,
            'misclassified_counts': misclassified_counts,
            'category_stats': category_stats,
            'misclassification_patterns': misclassification_patterns,
            'sample_misclassifications': misclassified.head(10).to_dict('records') if len(misclassified) > 0 else []
        }
