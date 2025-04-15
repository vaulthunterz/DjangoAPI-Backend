"""
Expense AI Service Module

This module provides AI functionality for expense categorization and prediction.
"""
import os
import logging
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional, Union

from .service import AIService
from expenses.ml.model_training import ExpenseCategoryClassifier
from expenses.ml.utils import load_training_data, validate_transaction_data

# Configure logging
logger = logging.getLogger(__name__)

class ExpenseAIService(AIService):
    """
    AI service for expense categorization and prediction.

    This service provides functionality for:
    - Predicting expense categories based on transaction descriptions
    - Training custom expense categorization models
    - Evaluating model performance
    """

    def __init__(self):
        """Initialize the Expense AI service."""
        super().__init__()
        self.classifier = None
        self.custom_model = None
        self.custom_vectorizer = None
        self.model_path = os.path.join('expenses', 'ml')

    def initialize(self) -> bool:
        """
        Initialize the Expense AI service and load required models.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Initialize the main classifier
            self.classifier = ExpenseCategoryClassifier()
            loaded = self.classifier.load_model()

            if not loaded:
                logger.warning("Main expense classifier model not found. Will be trained on first use.")

            # Try to load the custom model if it exists
            try:
                custom_model_path = os.path.join(self.model_path, 'custom_model.joblib')
                vectorizer_path = os.path.join(self.model_path, 'vectorizer.joblib')

                if os.path.exists(custom_model_path) and os.path.exists(vectorizer_path):
                    self.custom_model = joblib.load(custom_model_path)
                    self.custom_vectorizer = joblib.load(vectorizer_path)
                    logger.info("Custom expense model loaded successfully")
                else:
                    logger.info("Custom expense model not found")
            except Exception as e:
                logger.error(f"Error loading custom expense model: {str(e)}")

            self.models = {
                "main_classifier": self.classifier is not None,
                "custom_model": self.custom_model is not None
            }

            self.initialized = True
            logger.info("Expense AI service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing Expense AI service: {str(e)}")
            self.initialized = False
            return False

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the category and subcategory for a transaction.

        Args:
            data: Dictionary containing transaction information
                 Required keys: 'description', 'merchant' (optional)

        Returns:
            Dict[str, Any]: Prediction results with category, subcategory, and confidence scores
        """
        self._ensure_initialized()

        # Validate input data
        if 'description' not in data or not data['description']:
            raise ValueError("Transaction description is required for prediction")

        # Ensure the classifier is loaded
        if self.classifier is None:
            logger.warning("Classifier not loaded. Training with default data.")
            self._train_default_model()

        # Make prediction
        try:
            prediction = self.classifier.predict(data)
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

    def predict_with_custom_model(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the category using the custom trained model.

        Args:
            data: Dictionary containing transaction information
                 Required keys: 'description'

        Returns:
            Dict[str, Any]: Prediction results with category and confidence score
        """
        self._ensure_initialized()

        # Validate input data
        if 'description' not in data or not data['description']:
            raise ValueError("Transaction description is required for prediction")

        # Check if custom model is available
        if self.custom_model is None or self.custom_vectorizer is None:
            raise ValueError("Custom model not available. Train the model first.")

        try:
            # Vectorize the input
            description = data['description']
            X = self.custom_vectorizer.transform([description])

            # Make prediction
            predicted_label = self.custom_model.predict(X)[0]
            probabilities = self.custom_model.predict_proba(X)[0]
            confidence = float(max(probabilities))

            # Check if the prediction is a category name or ID
            from expenses.models import Category, SubCategory

            try:
                # If it's a category ID (integer)
                if isinstance(predicted_label, (int, float)) or (isinstance(predicted_label, str) and predicted_label.isdigit()):
                    category_id = int(predicted_label)
                    category = Category.objects.get(id=category_id)
                    category_name = category.name

                    # Try to get a default subcategory
                    try:
                        subcategory = SubCategory.objects.filter(category=category).first()
                        subcategory_id = subcategory.id if subcategory else None
                        subcategory_name = subcategory.name if subcategory else None
                    except Exception:
                        subcategory_id = None
                        subcategory_name = None

                # If it's a category name (string)
                else:
                    # Check if it's in "Category - Subcategory" format
                    if isinstance(predicted_label, str) and " - " in predicted_label:
                        parts = predicted_label.split(" - ")
                        category_name = parts[0]
                        subcategory_name = parts[1] if len(parts) > 1 else None

                        # Get category and subcategory IDs
                        category = Category.objects.get(name=category_name)
                        category_id = category.id

                        if subcategory_name:
                            subcategory = SubCategory.objects.get(name=subcategory_name, category=category)
                            subcategory_id = subcategory.id
                        else:
                            subcategory_id = None
                    else:
                        # Just a category name
                        category_name = predicted_label
                        category = Category.objects.get(name=category_name)
                        category_id = category.id
                        subcategory_id = None
                        subcategory_name = None

            except (Category.DoesNotExist, SubCategory.DoesNotExist, ValueError) as e:
                # If category doesn't exist, use Unknown
                logger.warning(f"Error finding category: {str(e)}. Using Unknown category.")
                try:
                    category = Category.objects.get(name='Unknown')
                    category_id = category.id
                    category_name = 'Unknown'

                    subcategory = SubCategory.objects.get(name='Other', category=category)
                    subcategory_id = subcategory.id
                    subcategory_name = 'Other'
                except (Category.DoesNotExist, SubCategory.DoesNotExist):
                    # If Unknown category doesn't exist, return the prediction as is
                    return {
                        'category': str(predicted_label),
                        'confidence': confidence
                    }

            return {
                'category': category_name,
                'subcategory': subcategory_name,
                'category_id': category_id,
                'subcategory_id': subcategory_id,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error making custom model prediction: {str(e)}")
            raise

    def train(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the expense categorization model with the provided data.

        Args:
            data: DataFrame containing training data. If None, loads default training data.

        Returns:
            Dict[str, Any]: Training results with performance metrics
        """
        self._ensure_initialized()

        try:
            # If no data provided, load default training data
            if data is None:
                data = load_training_data()

            # Validate the data
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Invalid training data")

            # Train the model
            if self.classifier is None:
                self.classifier = ExpenseCategoryClassifier()

            results = self.classifier.train_model(data)
            logger.info(f"Model trained successfully. Accuracy: {results.get('category_accuracy', 0):.2f}")

            return results

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def train_custom_model(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train a custom expense categorization model with the provided transactions.

        Args:
            transactions: List of transaction dictionaries

        Returns:
            Dict[str, Any]: Training results with performance metrics
        """
        self._ensure_initialized()

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics import classification_report, confusion_matrix
            import numpy as np

            # Convert to DataFrame
            df = pd.DataFrame(transactions)

            # Validate required columns
            required_columns = ['description', 'category']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Training data must contain: {', '.join(required_columns)}")

            # Prepare data
            X = df['description'].values
            y = df['category'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_vec, y_train)

            # Calculate metrics
            accuracy = clf.score(X_test_vec, y_test)
            y_pred = clf.predict(X_test_vec)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()

            # Get feature importance
            feature_importance = dict(zip(vectorizer.get_feature_names_out(), clf.feature_importances_))

            # Save model components
            os.makedirs(self.model_path, exist_ok=True)
            joblib.dump(clf, os.path.join(self.model_path, 'custom_model.joblib'))
            joblib.dump(vectorizer, os.path.join(self.model_path, 'vectorizer.joblib'))

            # Update instance variables
            self.custom_model = clf
            self.custom_vectorizer = vectorizer

            # Return metrics
            return {
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': conf_matrix,
                'feature_importance': feature_importance
            }

        except Exception as e:
            logger.error(f"Error training custom model: {str(e)}")
            raise

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the trained models.

        Returns:
            Dict[str, Any]: Model performance metrics
        """
        self._ensure_initialized()

        metrics = {
            'main_model': None,
            'custom_model': None
        }

        # Get metrics for main model if available
        if self.classifier and hasattr(self.classifier, 'get_metrics'):
            metrics['main_model'] = self.classifier.get_metrics()

        # Get metrics for custom model if available
        if self.custom_model:
            # For custom model, we can only provide basic info
            metrics['custom_model'] = {
                'available': True,
                'feature_count': len(self.custom_vectorizer.get_feature_names_out()) if self.custom_vectorizer else 0,
                'model_type': type(self.custom_model).__name__
            }

        return metrics

    def _train_default_model(self) -> None:
        """Train the model with default training data."""
        logger.info("Training model with default data")
        data = load_training_data()
        self.train(data)
