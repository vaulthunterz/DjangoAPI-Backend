"""
Expense AI Service Module

This module provides AI functionality for expense categorization and prediction.
"""
import os
import logging
import pandas as pd
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Union

from .service import AIService
from expenses.ml.core.model import ExpenseCategorizer
from expenses.ml.config import FEATURE_WEIGHTS, get_models_dir, log_config_settings
from expenses.ml.utils.data_utils import load_training_data, validate_transaction_data

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
            # Initialize the main classifier with feature weights from config
            models_dir = os.path.join(self.model_path, 'models')
            # Log the current configuration
            log_config_settings()
            self.classifier = ExpenseCategorizer(
                models_dir=models_dir,
                # Feature weights will be loaded from config
            )

            try:
                self.classifier.load_models()
                loaded = True
            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                loaded = False

            if not loaded:
                logger.warning("Main expense classifier model not found. Will be trained on first use.")

            # Load the trained model as the custom model
            try:
                from .settings import EXPENSE_CUSTOM_MODEL_PATH, EXPENSE_VECTORIZER_PATH

                if os.path.exists(EXPENSE_CUSTOM_MODEL_PATH) and os.path.exists(EXPENSE_VECTORIZER_PATH):
                    self.custom_model = joblib.load(EXPENSE_CUSTOM_MODEL_PATH)
                    self.custom_vectorizer = joblib.load(EXPENSE_VECTORIZER_PATH)
                    logger.info("Custom expense model loaded successfully from trained model")
                else:
                    logger.info("Custom expense model not found in models directory")
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
            # Convert data to the format expected by ExpenseCategorizer
            transaction = {
                'description': data.get('description', ''),
                'merchant': data.get('merchant', ''),
                'amount': data.get('amount', 0)
            }

            prediction = self.classifier.predict(transaction)

            # Add category_id and subcategory_id if needed
            if 'category' in prediction and prediction['category']:
                try:
                    from expenses.models import Category, SubCategory

                    # Try to get category ID
                    try:
                        category = Category.objects.get(name=prediction['category'])
                        prediction['category_id'] = category.id
                    except Category.DoesNotExist:
                        prediction['category_id'] = None

                    # Try to get subcategory ID
                    if 'subcategory' in prediction and prediction['subcategory']:
                        try:
                            subcategory = SubCategory.objects.get(
                                name=prediction['subcategory'],
                                category=category
                            )
                            prediction['subcategory_id'] = subcategory.id
                        except SubCategory.DoesNotExist:
                            prediction['subcategory_id'] = None
                except Exception as e:
                    logger.warning(f"Error getting category/subcategory IDs: {str(e)}")

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
            # Prepare input data for the vectorizer
            description = data['description']
            merchant = data.get('merchant', '')
            # Removed amount as it's not used in the model

            logger.info(f"Input description: {description}")
            logger.info(f"Input merchant: {merchant}")
            logger.info(f"Custom vectorizer type: {type(self.custom_vectorizer)}")

            # Create a DataFrame with ONLY the input data columns that were present during training
            # Based on the error message, 'amount' was not in the training data
            input_df = pd.DataFrame({
                'description': [description],
                'merchant': [merchant]
                # Removed 'amount' as it wasn't in the training data
            })

            try:
                # Transform using the feature extractor
                X = self.custom_vectorizer.transform(input_df)
                logger.info(f"Transformed input shape: {X.shape}")
            except Exception as transform_error:
                logger.error(f"Error transforming input: {str(transform_error)}")
                # Fallback to simpler approach if transform fails
                logger.info("Using fallback approach for prediction")

                # Use the regular predict method instead
                if self.classifier:
                    transaction = {
                        'description': description,
                        'merchant': merchant
                        # Removed 'amount' as it wasn't in the training data
                    }
                    prediction = self.classifier.predict(transaction)

                    return {
                        'category': prediction.get('category', 'Unknown'),
                        'subcategory': prediction.get('subcategory', 'Other'),
                        'confidence': prediction.get('category_confidence', 0.85),
                        'category_id': None,  # Will be looked up later
                        'subcategory_id': None  # Will be looked up later
                    }
                else:
                    raise ValueError("Both custom model and classifier failed")

            # Make prediction
            logger.info(f"Custom model type: {type(self.custom_model)}")

            try:
                # Get the prediction - handle different model types
                if hasattr(self.custom_model, 'predict'):
                    raw_prediction = self.custom_model.predict(X)
                    logger.info(f"Raw prediction: {raw_prediction}, type: {type(raw_prediction)}")

                    # Handle different prediction formats
                    if isinstance(raw_prediction, (list, np.ndarray)) and len(raw_prediction) > 0:
                        predicted_label = raw_prediction[0]
                    else:
                        predicted_label = raw_prediction

                    logger.info(f"Processed predicted label: {predicted_label}, type: {type(predicted_label)}")

                    # Get probabilities if available
                    if hasattr(self.custom_model, 'predict_proba'):
                        try:
                            raw_probabilities = self.custom_model.predict_proba(X)
                            logger.info(f"Raw probabilities: {raw_probabilities}, type: {type(raw_probabilities)}")

                            if isinstance(raw_probabilities, (list, np.ndarray)) and len(raw_probabilities) > 0:
                                probabilities = raw_probabilities[0]
                            else:
                                probabilities = raw_probabilities

                            logger.info(f"Processed probabilities: {probabilities}, type: {type(probabilities)}")

                            # Calculate confidence
                            if isinstance(probabilities, (list, np.ndarray)) and len(probabilities) > 0:
                                confidence = float(max(probabilities))
                            else:
                                confidence = 0.85  # Default confidence if we can't calculate it
                        except Exception as prob_error:
                            logger.error(f"Error getting probabilities: {str(prob_error)}")
                            confidence = 0.85  # Default confidence
                    else:
                        logger.warning("Model doesn't have predict_proba method, using default confidence")
                        confidence = 0.85  # Default confidence
                else:
                    logger.error("Custom model doesn't have predict method")
                    raise ValueError("Invalid custom model - no predict method")

            except Exception as pred_error:
                logger.error(f"Error during prediction: {str(pred_error)}")
                raise

            logger.info(f"Final prediction: {predicted_label}, confidence: {confidence}")

            # Check if the prediction is a category name or ID
            from expenses.models import Category, SubCategory

            logger.info(f"Processing predicted label: {predicted_label}")

            try:
                # Get all categories for fallback
                all_categories = list(Category.objects.all())
                logger.info(f"Found {len(all_categories)} categories in database")

                # If it's a category ID (integer)
                if isinstance(predicted_label, (int, float)) or (isinstance(predicted_label, str) and str(predicted_label).isdigit()):
                    logger.info(f"Treating prediction as category ID")
                    category_id = int(predicted_label)
                    logger.info(f"Looking up category with ID: {category_id}")

                    try:
                        category = Category.objects.get(id=category_id)
                        category_name = category.name
                        logger.info(f"Found category: {category_name}")
                    except Category.DoesNotExist:
                        logger.warning(f"Category with ID {category_id} not found, using first available category")
                        if all_categories:
                            category = all_categories[0]
                            category_id = category.id
                            category_name = category.name
                            logger.info(f"Using fallback category: {category_name}")
                        else:
                            raise ValueError("No categories found in database")

                    # Try to get a default subcategory
                    try:
                        subcategory = SubCategory.objects.filter(category=category).first()
                        subcategory_id = subcategory.id if subcategory else None
                        subcategory_name = subcategory.name if subcategory else None
                        logger.info(f"Found subcategory: {subcategory_name}")
                    except Exception as e:
                        logger.error(f"Error finding subcategory: {str(e)}")
                        subcategory_id = None
                        subcategory_name = None

                # If it's a category name (string)
                else:
                    logger.info(f"Treating prediction as category name")

                    # Convert to string if it's not already
                    if not isinstance(predicted_label, str):
                        predicted_label = str(predicted_label)
                        logger.info(f"Converted prediction to string: {predicted_label}")

                    # Check if it's in "Category - Subcategory" format
                    if " - " in predicted_label:
                        logger.info(f"Prediction contains category and subcategory")
                        parts = predicted_label.split(" - ")
                        category_name = parts[0]
                        subcategory_name = parts[1] if len(parts) > 1 else None
                        logger.info(f"Split into category: {category_name}, subcategory: {subcategory_name}")

                        # Get category and subcategory IDs
                        logger.info(f"Looking up category with name: {category_name}")
                        try:
                            category = Category.objects.get(name=category_name)
                            category_id = category.id
                            logger.info(f"Found category ID: {category_id}")
                        except Category.DoesNotExist:
                            logger.warning(f"Category '{category_name}' not found, using first available category")
                            if all_categories:
                                category = all_categories[0]
                                category_id = category.id
                                category_name = category.name
                                logger.info(f"Using fallback category: {category_name}")
                            else:
                                raise ValueError("No categories found in database")

                        if subcategory_name:
                            logger.info(f"Looking up subcategory with name: {subcategory_name}")
                            try:
                                subcategory = SubCategory.objects.get(name=subcategory_name, category=category)
                                subcategory_id = subcategory.id
                                logger.info(f"Found subcategory ID: {subcategory_id}")
                            except SubCategory.DoesNotExist:
                                logger.warning(f"Subcategory '{subcategory_name}' not found, using first available subcategory")
                                subcategory = SubCategory.objects.filter(category=category).first()
                                if subcategory:
                                    subcategory_id = subcategory.id
                                    subcategory_name = subcategory.name
                                    logger.info(f"Using fallback subcategory: {subcategory_name}")
                                else:
                                    subcategory_id = None
                                    subcategory_name = None
                        else:
                            subcategory_id = None
                    else:
                        # Just a category name
                        logger.info(f"Prediction is just a category name")
                        category_name = predicted_label
                        logger.info(f"Looking up category with name: {category_name}")

                        try:
                            category = Category.objects.get(name=category_name)
                            category_id = category.id
                            logger.info(f"Found category ID: {category_id}")
                        except Category.DoesNotExist:
                            # Try to find a category that contains this text
                            logger.warning(f"Category '{category_name}' not found, trying partial match")
                            matching_categories = [c for c in all_categories if category_name.lower() in c.name.lower()]

                            if matching_categories:
                                category = matching_categories[0]
                                category_id = category.id
                                category_name = category.name
                                logger.info(f"Found partial match category: {category_name}")
                            else:
                                logger.warning(f"No matching category found, using first available category")
                                if all_categories:
                                    category = all_categories[0]
                                    category_id = category.id
                                    category_name = category.name
                                    logger.info(f"Using fallback category: {category_name}")
                                else:
                                    raise ValueError("No categories found in database")

                        subcategory_id = None
                        subcategory_name = None

                        # Try to find a default subcategory
                        subcategory = SubCategory.objects.filter(category=category).first()
                        if subcategory:
                            subcategory_id = subcategory.id
                            subcategory_name = subcategory.name
                            logger.info(f"Using default subcategory: {subcategory_name}")

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

            # Save data to a temporary CSV file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', newline='') as temp_file:
                data.to_csv(temp_file.name, index=False)
                temp_path = temp_file.name

            # Initialize the model if needed with weighted features
            models_dir = os.path.join(self.model_path, 'models')
            if self.classifier is None:
                self.classifier = ExpenseCategorizer(
                    models_dir=models_dir
                    # Feature weights will be loaded from config
                )

            # Train the model
            results = self.classifier.train(data_path=temp_path)

            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")

            logger.info(f"Model trained successfully. Accuracy: {results.get('category_accuracy', 0):.2f}")

            # Reload the custom model after training
            try:
                from .settings import EXPENSE_CUSTOM_MODEL_PATH, EXPENSE_VECTORIZER_PATH

                if os.path.exists(EXPENSE_CUSTOM_MODEL_PATH) and os.path.exists(EXPENSE_VECTORIZER_PATH):
                    self.custom_model = joblib.load(EXPENSE_CUSTOM_MODEL_PATH)
                    self.custom_vectorizer = joblib.load(EXPENSE_VECTORIZER_PATH)
                    logger.info("Custom expense model reloaded successfully after training")
                else:
                    logger.warning("Custom model files not found after training")
            except Exception as e:
                logger.error(f"Error reloading custom model after training: {str(e)}")

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
        if self.classifier:
            try:
                # Try to load performance metrics from file
                models_dir = os.path.join(self.model_path, 'models')
                performance_path = os.path.join(models_dir, 'performance.joblib')

                if os.path.exists(performance_path):
                    performance = joblib.load(performance_path)
                    metrics['main_model'] = performance
                else:
                    # Fallback to basic info
                    metrics['main_model'] = {
                        'available': True,
                        'model_type': 'ExpenseCategorizer'
                    }
            except Exception as e:
                logger.error(f"Error loading model metrics: {str(e)}")
                metrics['main_model'] = {
                    'available': True,
                    'error': str(e)
                }

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

        # Get the path to the default training data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        data_path = os.path.join(base_dir, 'expenses', 'ml', 'training_data', 'transactions.csv')

        # Initialize the model if needed with weighted features
        models_dir = os.path.join(self.model_path, 'models')
        if self.classifier is None:
            self.classifier = ExpenseCategorizer(
                models_dir=models_dir
                # Feature weights will be loaded from config
            )

        # Train the model directly with the data path
        if os.path.exists(data_path):
            logger.info(f"Training with data from {data_path}")
            self.classifier.train(data_path=data_path)

            # Reload the custom model after training
            try:
                from .settings import EXPENSE_CUSTOM_MODEL_PATH, EXPENSE_VECTORIZER_PATH

                if os.path.exists(EXPENSE_CUSTOM_MODEL_PATH) and os.path.exists(EXPENSE_VECTORIZER_PATH):
                    self.custom_model = joblib.load(EXPENSE_CUSTOM_MODEL_PATH)
                    self.custom_vectorizer = joblib.load(EXPENSE_VECTORIZER_PATH)
                    logger.info("Custom expense model reloaded successfully after training")
                else:
                    logger.warning("Custom model files not found after training")
            except Exception as e:
                logger.error(f"Error reloading custom model after training: {str(e)}")
        else:
            # Fallback to loading data and using the train method
            logger.warning(f"Default training data not found at {data_path}. Using fallback method.")
            data = load_training_data()
            self.train(data)
