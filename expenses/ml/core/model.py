"""
Ensemble model for expense categorization
Combines multiple models trained on different feature sets
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
import time
import sys
from tqdm import tqdm

from .features import create_enhanced_features
from ..config import (
    FEATURE_WEIGHTS, TEST_SIZE, RANDOM_STATE, USE_ENSEMBLE, N_JOBS,
    RF_CATEGORY_PARAMS, GB_CATEGORY_PARAMS, LR_CATEGORY_PARAMS,
    RF_SUBCATEGORY_PARAMS, GB_SUBCATEGORY_PARAMS,
    CATEGORY_ENSEMBLE_WEIGHTS, SUBCATEGORY_ENSEMBLE_WEIGHTS,
    EVALUATION_METRICS, TOP_FEATURES_COUNT,
    log_config_settings
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier that combines multiple models"""

    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights
        self.classes_ = None

    def fit(self, X, y):
        """Fit each model in the ensemble"""
        # Handle NaN values by replacing them with zeros
        X_clean = np.nan_to_num(X, nan=0.0)

        self.classes_ = np.unique(y)

        for model in self.models:
            try:
                model.fit(X_clean, y)
            except Exception as e:
                logger.error(f"Error fitting model {type(model).__name__}: {str(e)}")
                # Try to continue with other models

        # If weights not provided, use equal weights
        if self.weights is None:
            self.weights = np.ones(len(self.models)) / len(self.models)

        return self

    def predict_proba(self, X):
        """Predict class probabilities using weighted average of models"""
        # Handle NaN values by replacing them with zeros
        X_clean = np.nan_to_num(X, nan=0.0)

        # Collect probabilities from each model
        all_probas = []
        for model in self.models:
            try:
                probas = model.predict_proba(X_clean)
                all_probas.append(probas)
            except Exception as e:
                logger.error(f"Error in predict_proba for model {type(model).__name__}: {str(e)}")
                # Skip this model

        if not all_probas:
            # If no models could predict, return zeros
            return np.zeros((X.shape[0], len(self.classes_)))

        # Combine probabilities using weights
        probas = np.array(all_probas)
        return np.tensordot(self.weights[:len(all_probas)], probas, axes=([0], [0]))

    def predict(self, X):
        """Predict class using weighted voting"""
        proba = self.predict_proba(X)
        if proba.shape[1] > 0:
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Fallback if no valid predictions
            return np.array([self.classes_[0]] * X.shape[0])


class ExpenseCategorizer:
    """Main class for expense categorization using ensemble models"""

    def __init__(self, models_dir='models', use_ensemble=None, feature_weights=None):
        """
        Initialize the expense categorizer

        Args:
            models_dir (str): Directory to save/load models
            use_ensemble (bool): Whether to use ensemble model (True) or just RandomForest (False)
            feature_weights (dict): Weights to apply to different features
        """
        self.models_dir = models_dir
        self.category_model = None
        self.subcategory_models = {}
        self.feature_extractor = None
        self.categories = None
        self.subcategories = {}

        # Use values from config if not provided
        self.use_ensemble = use_ensemble if use_ensemble is not None else USE_ENSEMBLE

        # Use feature weights from config if not provided
        self.feature_weights = feature_weights or FEATURE_WEIGHTS.copy()

        # For backward compatibility
        self.description_weight = self.feature_weights.get('description', 10.0)
        self.merchant_weight = self.feature_weights.get('merchant', 0.0)

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Log configuration settings
        logger.info(f"Initializing ExpenseCategorizer with feature weights: {self.feature_weights}")

    def train(self, data_path, test_size=None, random_state=None):
        """
        Train the ensemble models

        Args:
            data_path: Path to the training data CSV file
            test_size: Fraction of data to use for testing (default from config)
            random_state: Random seed for reproducibility (default from config)
        """
        # Use values from config if not provided
        test_size = test_size if test_size is not None else TEST_SIZE
        random_state = random_state if random_state is not None else RANDOM_STATE

        # Start timing
        start_time = time.time()

        # Log configuration settings
        log_config_settings()

        # Print header for progress tracking
        print("\n" + "="*70)
        print(" ENHANCED EXPENSE CATEGORIZATION MODEL TRAINING")
        print("="*70)

        # Step 1: Load data
        print("\n[1/5] Loading training data...")
        df = pd.read_csv(data_path)
        print(f"      Loaded {len(df)} transactions with {df['category'].nunique()} categories")

        # Step 2: Create enhanced features with weights from config
        print("\n[2/5] Creating enhanced features...")
        feature_start = time.time()
        sys.stdout.write(f"      Extracting features (weights: {self.feature_weights})... ")
        sys.stdout.flush()
        X, self.feature_extractor = create_enhanced_features(
            df,
            feature_weights=self.feature_weights
        )

        # Check for NaN values in features
        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            print(f"\n      WARNING: Found {nan_count} NaN values in features. Filling with zeros.")
            X = X.fillna(0)

        feature_time = time.time() - feature_start
        print(f"Done! ({feature_time:.2f}s)")
        print(f"      Created {X.shape[1]} features from description and merchant text")

        y_category = df['category']

        # Store unique categories and subcategories
        print("\n[3/5] Analyzing category structure...")
        self.categories = y_category.unique()
        category_counts = df['category'].value_counts()

        print(f"      Found {len(self.categories)} unique categories")

        # Show top 5 most common categories
        print("\n      Top 5 most common categories:")
        for i, (category, count) in enumerate(category_counts.head(5).items()):
            print(f"        {i+1}. {category}: {count} transactions ({count/len(df)*100:.1f}%)")

        # Process subcategories with progress bar
        print("\n      Processing subcategories...")
        subcategory_counts = 0
        for category in tqdm(self.categories, desc="      ", ncols=70):
            category_mask = df['category'] == category
            subcats = df.loc[category_mask, 'subcategory'].unique()
            self.subcategories[category] = subcats
            subcategory_counts += len(subcats)

        print(f"      Found {subcategory_counts} total subcategories across all categories")

        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_category, test_size=test_size, random_state=random_state, stratify=y_category
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {str(e)}. Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_category, test_size=test_size, random_state=random_state
            )

        # Step 4: Train category model
        print("\n[4/5] Training main category model...")
        model_start = time.time()

        print("      Creating ensemble model...")
        self.category_model = self._create_category_ensemble()

        print("      Fitting model on training data...")
        # Set verbose=1 for RandomForest progress
        if hasattr(self.category_model, 'models'):
            # For ensemble models
            for model in self.category_model.models:
                if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
                    model.verbose = 1
                elif isinstance(model, Pipeline):
                    # For Pipeline objects, we need to set verbose on the classifier
                    if hasattr(model, 'steps'):
                        for name, step in model.steps:
                            if hasattr(step, 'verbose'):
                                step.verbose = 1
        else:
            # For non-ensemble models (Pipeline)
            if isinstance(self.category_model, Pipeline):
                for name, step in self.category_model.steps:
                    if hasattr(step, 'verbose'):
                        step.verbose = 1
            # For direct models
            elif hasattr(self.category_model, 'verbose'):
                self.category_model.verbose = 1

        self.category_model.fit(X_train, y_train)

        # Evaluate category model
        print("      Evaluating model on test data...")
        y_pred = self.category_model.predict(X_test)
        category_accuracy = accuracy_score(y_test, y_pred)

        # Calculate precision, recall, and f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        # Evaluate individual models if available
        individual_model_metrics = []
        if hasattr(self.category_model, 'models'):
            print("      Evaluating individual models...")
            for i, model in enumerate(self.category_model.models):
                try:
                    model_name = type(model).__name__
                    if isinstance(model, Pipeline):
                        for name, step in model.steps:
                            if hasattr(step, 'predict'):
                                model_name = type(step).__name__
                                break

                    # Get predictions from this model
                    if hasattr(model, 'predict'):
                        model_preds = model.predict(X_test)
                        model_accuracy = accuracy_score(y_test, model_preds)
                        model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(
                            y_test, model_preds, average='weighted'
                        )

                        individual_model_metrics.append({
                            'model_name': model_name,
                            'accuracy': float(model_accuracy),
                            'precision': float(model_precision),
                            'recall': float(model_recall),
                            'f1_score': float(model_f1)
                        })

                        print(f"        - {model_name}: Accuracy={model_accuracy*100:.2f}%, F1={model_f1*100:.2f}%")
                except Exception as e:
                    logger.error(f"Error evaluating individual model {i}: {str(e)}")

        model_time = time.time() - model_start
        print(f"      ✓ Category model trained in {model_time:.2f}s")
        print(f"      ✓ Accuracy: {category_accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%")

        # Step 5: Train subcategory models
        print("\n[5/5] Training subcategory models...")
        subcategory_accuracies = {}

        # Count valid categories for progress tracking
        valid_categories = []
        for category in self.categories:
            category_mask = df['category'] == category
            if category_mask.sum() < 2:
                continue

            category_df = df[category_mask]
            y_subcategory = category_df['subcategory']

            if len(y_subcategory.unique()) < 2:
                continue

            valid_categories.append(category)

        print(f"      Training {len(valid_categories)} subcategory models...")

        # Create progress bar for subcategory models
        for i, category in enumerate(tqdm(valid_categories, desc="      ", ncols=70)):
            # Filter data for this category
            category_mask = df['category'] == category
            category_df = df[category_mask]

            # Extract features for this category with weights from config
            X_cat, _ = create_enhanced_features(
                category_df,
                feature_weights=self.feature_weights
            )

            # Check for NaN values in subcategory features
            nan_count = X_cat.isna().sum().sum()
            if nan_count > 0:
                print(f"        WARNING: Found {nan_count} NaN values in features for {category}. Filling with zeros.")
                X_cat = X_cat.fillna(0)

            y_subcategory = category_df['subcategory']

            # Split data
            try:
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                    X_cat, y_subcategory, test_size=test_size, random_state=random_state, stratify=y_subcategory
                )
            except ValueError:
                # If stratify fails, try without it
                X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
                    X_cat, y_subcategory, test_size=test_size, random_state=random_state
                )

            # Check if we have at least 2 classes in the training data
            if len(np.unique(y_train_sub)) < 2:
                print(f"        Skipping {category}: only one subcategory class in training data")
                continue

            try:
                # Train subcategory model
                subcategory_model = self._create_subcategory_ensemble()
                subcategory_model.fit(X_train_sub, y_train_sub)

                # Evaluate subcategory model
                y_pred_sub = subcategory_model.predict(X_test_sub)
                subcategory_accuracy = accuracy_score(y_test_sub, y_pred_sub)
                subcategory_accuracies[category] = subcategory_accuracy

                # Save subcategory model
                self.subcategory_models[category] = subcategory_model
            except Exception as e:
                print(f"        Error training model for {category}: {str(e)}")
                continue

        # Calculate total training time
        total_time = time.time() - start_time

        # Create performance metrics
        performance_metrics = {
            'category_accuracy': category_accuracy,
            'category_precision': precision,
            'category_recall': recall,
            'category_f1': f1,
            'individual_model_metrics': individual_model_metrics,
            'subcategory_accuracies': subcategory_accuracies,
            'training_time': total_time
        }

        # Save performance metrics
        try:
            joblib.dump(performance_metrics, os.path.join(self.models_dir, 'performance.joblib'))
            logger.info(f"Performance metrics saved to {os.path.join(self.models_dir, 'performance.joblib')}")
        except Exception as e:
            logger.error(f"Error saving performance metrics: {str(e)}")

        # Save models and metadata
        print("\n      Saving models and metadata...")
        self._save_models()

        # Print summary
        print("\n" + "="*70)
        print(" TRAINING COMPLETE")
        print("="*70)
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Category model accuracy: {category_accuracy*100:.2f}%")

        # Calculate average subcategory accuracy
        avg_subcategory_accuracy = sum(subcategory_accuracies.values()) / len(subcategory_accuracies) if subcategory_accuracies else 0
        print(f"Average subcategory accuracy: {avg_subcategory_accuracy*100:.2f}%")

        # Show top 5 best performing subcategory models
        if subcategory_accuracies:
            print("\nTop 5 best performing subcategory models:")
            top_subcategories = sorted(subcategory_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (category, accuracy) in enumerate(top_subcategories):
                print(f"  {i+1}. {category}: {accuracy*100:.2f}%")

        print("\nModel training complete! You can now use this model for predictions.")

        return {
            'category_accuracy': category_accuracy,
            'category_precision': precision,
            'category_recall': recall,
            'category_f1': f1,
            'individual_model_metrics': individual_model_metrics,
            'subcategory_accuracies': subcategory_accuracies,
            'training_time': total_time
        }

    def predict(self, transaction):
        """Predict category and subcategory for a transaction"""
        if not self.category_model:
            raise ValueError("Models not trained or loaded")

        # Convert transaction to DataFrame
        if isinstance(transaction, dict):
            transaction = pd.DataFrame([transaction])

        try:
            # Extract features using the same feature extractor used during training
            if self.feature_extractor:
                # Use the saved feature extractor
                X = self.feature_extractor.transform(transaction)
            else:
                # Fallback to creating new features with weights from config
                X, _ = create_enhanced_features(
                    transaction,
                    feature_weights=self.feature_weights
                )

            # Predict category with confidence scores
            # Handle both ensemble and non-ensemble models
            if hasattr(self.category_model, 'predict_proba'):
                category_probs = self.category_model.predict_proba(X)[0]
                category_idx = np.argmax(category_probs)

                # Get the classes attribute from the appropriate location
                if hasattr(self.category_model, 'classes_'):
                    classes = self.category_model.classes_
                elif hasattr(self.category_model, 'steps') and hasattr(self.category_model[-1], 'classes_'):
                    # For Pipeline objects
                    classes = self.category_model[-1].classes_
                else:
                    # Fallback to unique categories from training
                    classes = np.array(self.categories)

                category = classes[category_idx]
                category_confidence = float(category_probs[category_idx])

                # Get top 3 category predictions with confidence scores
                top_categories = []
                sorted_indices = np.argsort(category_probs)[::-1][:3]  # Top 3 indices
                for idx in sorted_indices:
                    if idx < len(classes):
                        top_categories.append({
                            'category': classes[idx],
                            'confidence': float(category_probs[idx])
                        })
            else:
                # Fallback for models without predict_proba
                category = self.category_model.predict(X)[0]
                category_confidence = 1.0  # Default confidence
                top_categories = [{'category': category, 'confidence': 1.0}]

            # Predict subcategory if we have a model for this category
            subcategory = None
            subcategory_confidence = 0.0
            top_subcategories = []

            if category in self.subcategory_models:
                subcategory_model = self.subcategory_models[category]

                # Handle both ensemble and non-ensemble models
                if hasattr(subcategory_model, 'predict_proba'):
                    subcategory_probs = subcategory_model.predict_proba(X)[0]
                    subcategory_idx = np.argmax(subcategory_probs)

                    # Get the classes attribute from the appropriate location
                    if hasattr(subcategory_model, 'classes_'):
                        classes = subcategory_model.classes_
                    elif hasattr(subcategory_model, 'steps') and hasattr(subcategory_model[-1], 'classes_'):
                        # For Pipeline objects
                        classes = subcategory_model[-1].classes_
                    else:
                        # Fallback to unique subcategories from training
                        classes = np.array(self.subcategories.get(category, []))

                    if len(classes) > 0:
                        subcategory = classes[subcategory_idx]
                        subcategory_confidence = float(subcategory_probs[subcategory_idx])

                        # Get top 3 subcategory predictions with confidence scores
                        sorted_indices = np.argsort(subcategory_probs)[::-1][:3]  # Top 3 indices
                        for idx in sorted_indices:
                            if idx < len(classes):
                                top_subcategories.append({
                                    'subcategory': classes[idx],
                                    'confidence': float(subcategory_probs[idx])
                                })
                else:
                    # Fallback for models without predict_proba
                    subcategory = subcategory_model.predict(X)[0]
                    subcategory_confidence = 1.0  # Default confidence
                    top_subcategories = [{'subcategory': subcategory, 'confidence': 1.0}]

            # Get individual model predictions for the category
            individual_model_predictions = []

            # For ensemble models, get predictions from each component model
            if hasattr(self.category_model, 'models'):
                for i, model in enumerate(self.category_model.models):
                    try:
                        if hasattr(model, 'predict_proba'):
                            model_probs = model.predict_proba(X)[0]
                            model_idx = np.argmax(model_probs)
                            model_class = model.classes_[model_idx] if hasattr(model, 'classes_') else self.category_model.classes_[model_idx]
                            model_confidence = float(model_probs[model_idx])

                            model_name = type(model).__name__
                            if isinstance(model, Pipeline):
                                for name, step in model.steps:
                                    if hasattr(step, 'predict_proba'):
                                        model_name = type(step).__name__
                                        break

                            individual_model_predictions.append({
                                'model': model_name,
                                'prediction': model_class,
                                'confidence': model_confidence
                            })
                    except Exception as e:
                        logger.error(f"Error getting individual model prediction: {str(e)}")
            else:
                # For non-ensemble models, just add the main model
                try:
                    model_name = type(self.category_model).__name__
                    if isinstance(self.category_model, Pipeline):
                        for name, step in self.category_model.steps:
                            if hasattr(step, 'predict'):
                                model_name = type(step).__name__
                                break

                    individual_model_predictions.append({
                        'model': model_name,
                        'prediction': category,
                        'confidence': category_confidence
                    })
                except Exception as e:
                    logger.error(f"Error getting main model prediction: {str(e)}")

            return {
                'category': category,
                'category_confidence': category_confidence,
                'subcategory': subcategory,
                'subcategory_confidence': subcategory_confidence,
                'top_categories': top_categories,
                'top_subcategories': top_subcategories,
                'individual_model_predictions': individual_model_predictions
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Fallback to a simpler prediction method
            logger.info("Using fallback prediction method...")

            # Get the most similar category based on text matching
            desc = transaction['description'].iloc[0].lower() if 'description' in transaction.columns else ''
            merch = transaction['merchant'].iloc[0].lower() if 'merchant' in transaction.columns else ''

            # Simple matching based on text similarity
            best_match = None
            best_score = -1

            for cat in self.categories:
                # Simple word matching score
                cat_lower = cat.lower()
                score = 0
                if cat_lower in desc or any(word in desc for word in cat_lower.split()):
                    score += 2
                if cat_lower in merch or any(word in merch for word in cat_lower.split()):
                    score += 3

                if score > best_score:
                    best_score = score
                    best_match = cat

            # If no match found, use the first category
            if best_match is None and len(self.categories) > 0:
                best_match = self.categories[0]

            return {
                'category': best_match,
                'category_confidence': 0.0,
                'subcategory': None,
                'subcategory_confidence': 0.0,
                'top_categories': [],
                'top_subcategories': [],
                'individual_model_predictions': []
            }

    def _create_category_ensemble(self):
        """Create ensemble model for category prediction"""
        # Create RandomForest model with imputation for missing values
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', RandomForestClassifier(**RF_CATEGORY_PARAMS))
        ])

        # If not using ensemble, return just the RandomForest model
        if not self.use_ensemble:
            logger.info("Using RandomForest model only (no ensemble)")
            return rf_pipeline

        # Otherwise, create the full ensemble with multiple models
        logger.info("Creating ensemble model with multiple classifiers")

        # Add GradientBoosting model
        gb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', GradientBoostingClassifier(**GB_CATEGORY_PARAMS))
        ])

        lr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**LR_CATEGORY_PARAMS))
        ])

        # Get weights from config
        weights = [
            CATEGORY_ENSEMBLE_WEIGHTS['random_forest'],
            CATEGORY_ENSEMBLE_WEIGHTS['gradient_boosting'],
            CATEGORY_ENSEMBLE_WEIGHTS['logistic_regression']
        ]

        # Create ensemble
        ensemble = EnsembleClassifier(
            models=[rf_pipeline, gb_pipeline, lr_pipeline],
            weights=weights  # Weights from config
        )

        return ensemble

    def _create_subcategory_ensemble(self):
        """Create ensemble model for subcategory prediction"""
        # Create RandomForest model with imputation for missing values
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', RandomForestClassifier(**RF_SUBCATEGORY_PARAMS))
        ])

        # If not using ensemble, return just the RandomForest model
        if not self.use_ensemble:
            logger.info("Using RandomForest model only for subcategories (no ensemble)")
            return rf_pipeline

        # Otherwise, create the full ensemble with multiple models
        logger.info("Creating ensemble model for subcategories")

        gb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', GradientBoostingClassifier(**GB_SUBCATEGORY_PARAMS))
        ])

        # Get weights from config
        weights = [
            SUBCATEGORY_ENSEMBLE_WEIGHTS['random_forest'],
            SUBCATEGORY_ENSEMBLE_WEIGHTS['gradient_boosting']
        ]

        # Create ensemble
        ensemble = EnsembleClassifier(
            models=[rf_pipeline, gb_pipeline],
            weights=weights  # Weights from config
        )

        return ensemble

    def _save_models(self):
        """Save trained models and metadata"""
        # Clean up old model files first
        logger.info(f"Cleaning up old model files in {self.models_dir}")
        try:
            # Remove old model files
            for file in os.listdir(self.models_dir):
                if (file.startswith('subcategory_model_') or
                    file in ['category_model.pkl', 'category_model.joblib',
                             'feature_extractor.pkl', 'feature_extractor.joblib',
                             'metadata.pkl', 'metadata.joblib']):
                    os.remove(os.path.join(self.models_dir, file))
                    logger.info(f"Removed old model file: {file}")
        except Exception as e:
            logger.error(f"Error cleaning up old model files: {str(e)}")

        # Save category model
        joblib.dump(self.category_model, os.path.join(self.models_dir, 'category_model.joblib'))

        # Save subcategory models
        for category, model in self.subcategory_models.items():
            safe_category = category.replace('/', '_').replace('\\', '_')
            joblib.dump(model, os.path.join(self.models_dir, f'subcategory_model_{safe_category}.joblib'))

        # Save feature extractor
        joblib.dump(self.feature_extractor, os.path.join(self.models_dir, 'feature_extractor.joblib'))

        # Save metadata with feature weights from config
        metadata = {
            'categories': self.categories,
            'subcategories': self.subcategories,
            'feature_weights': self.feature_weights,
            'description_weight': self.description_weight,  # For backward compatibility
            'merchant_weight': self.merchant_weight,        # For backward compatibility
            'model_version': '3.0',  # Increment version to track changes
            'feature_weighting_enabled': True,
            'use_ensemble': self.use_ensemble
        }
        joblib.dump(metadata, os.path.join(self.models_dir, 'metadata.joblib'))

        logger.info(f"Models and metadata saved to {self.models_dir}")

    def load_models(self):
        """Load trained models and metadata"""
        try:
            # Try loading joblib files first
            category_model_path = os.path.join(self.models_dir, 'category_model.joblib')
            metadata_path = os.path.join(self.models_dir, 'metadata.joblib')
            feature_extractor_path = os.path.join(self.models_dir, 'feature_extractor.joblib')

            # Check if joblib files exist
            if os.path.exists(category_model_path) and os.path.exists(metadata_path):
                # Load category model
                self.category_model = joblib.load(category_model_path)

                # Load metadata
                metadata = joblib.load(metadata_path)
                self.categories = metadata['categories']
                self.subcategories = metadata['subcategories']

                # Load feature weights if available (new format)
                if 'feature_weights' in metadata:
                    self.feature_weights = metadata['feature_weights']
                    logger.info(f"Loaded feature weights: {self.feature_weights}")

                    # Update individual weights for backward compatibility
                    self.description_weight = self.feature_weights.get('description', 10.0)
                    self.merchant_weight = self.feature_weights.get('merchant', 0.0)
                # Load individual weights (old format)
                elif 'description_weight' in metadata and 'merchant_weight' in metadata:
                    self.description_weight = metadata['description_weight']
                    self.merchant_weight = metadata['merchant_weight']
                    logger.info(f"Loaded legacy weights: description={self.description_weight}, merchant={self.merchant_weight}")

                    # Update feature_weights dictionary
                    self.feature_weights = {
                        'description': self.description_weight,
                        'merchant': self.merchant_weight
                    }

                # Load use_ensemble if available
                if 'use_ensemble' in metadata:
                    self.use_ensemble = metadata['use_ensemble']
                    logger.info(f"Loaded use_ensemble: {self.use_ensemble}")

                # Log model version if available
                if 'model_version' in metadata:
                    logger.info(f"Loaded model version: {metadata['model_version']}")
                if 'feature_weighting_enabled' in metadata:
                    logger.info(f"Feature weighting enabled: {metadata['feature_weighting_enabled']}")

                # Load subcategory models
                for category in self.categories:
                    safe_category = category.replace('/', '_').replace('\\', '_')
                    model_path = os.path.join(self.models_dir, f'subcategory_model_{safe_category}.joblib')
                    if os.path.exists(model_path):
                        self.subcategory_models[category] = joblib.load(model_path)

                # Load feature extractor
                if os.path.exists(feature_extractor_path):
                    self.feature_extractor = joblib.load(feature_extractor_path)

                logger.info(f"Models and metadata loaded from {self.models_dir} (joblib format)")
                return

            # Fallback to pickle files for backward compatibility
            logger.warning("Joblib files not found, trying pickle files for backward compatibility")
            import pickle

            # Load category model
            with open(os.path.join(self.models_dir, 'category_model.pkl'), 'rb') as f:
                self.category_model = pickle.load(f)

            # Load metadata
            with open(os.path.join(self.models_dir, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
                self.categories = metadata['categories']
                self.subcategories = metadata['subcategories']

            # Load subcategory models
            for category in self.categories:
                safe_category = category.replace('/', '_').replace('\\', '_')
                model_path = os.path.join(self.models_dir, f'subcategory_model_{safe_category}.pkl')
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.subcategory_models[category] = pickle.load(f)

            # Load feature extractor
            with open(os.path.join(self.models_dir, 'feature_extractor.pkl'), 'rb') as f:
                self.feature_extractor = pickle.load(f)

            logger.info(f"Models and metadata loaded from {self.models_dir} (pickle format)")

            # Convert to joblib format for future use
            logger.info("Converting pickle files to joblib format for future use")
            self._save_models()

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def _get_feature_importance(self):
        """Get feature importance from the category model with descriptive names"""
        # Only works for RandomForest and GradientBoosting models
        feature_importance = {}
        feature_descriptions = {}

        # Get feature descriptions if available
        if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_descriptions'):
            feature_descriptions = self.feature_extractor.get_feature_descriptions()

        # Handle both ensemble and non-ensemble models
        if hasattr(self.category_model, 'models'):
            # Ensemble model
            for model in self.category_model.models:
                if isinstance(model, Pipeline):
                    # For Pipeline objects, get the classifier
                    for name, step in model.steps:
                        if hasattr(step, 'feature_importances_'):
                            # Get feature names from feature extractor
                            if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                                feature_names = self.feature_extractor.get_feature_names_out()
                                for i, importance in enumerate(step.feature_importances_):
                                    if i < len(feature_names):
                                        feature_name = feature_names[i]
                                        if feature_name in feature_importance:
                                            feature_importance[feature_name] += float(importance)
                                        else:
                                            feature_importance[feature_name] = float(importance)
                            break
                elif hasattr(model, 'feature_importances_'):
                    # Direct model with feature_importances_
                    if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                        feature_names = self.feature_extractor.get_feature_names_out()
                        for i, importance in enumerate(model.feature_importances_):
                            if i < len(feature_names):
                                feature_name = feature_names[i]
                                if feature_name in feature_importance:
                                    feature_importance[feature_name] += float(importance)
                                else:
                                    feature_importance[feature_name] = float(importance)
        else:
            # Non-ensemble model (direct model or pipeline)
            if isinstance(self.category_model, Pipeline):
                # For Pipeline objects, get the classifier
                for name, step in self.category_model.steps:
                    if hasattr(step, 'feature_importances_'):
                        # Get feature names from feature extractor
                        if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                            feature_names = self.feature_extractor.get_feature_names_out()
                            for i, importance in enumerate(step.feature_importances_):
                                if i < len(feature_names):
                                    feature_name = feature_names[i]
                                    feature_importance[feature_name] = float(importance)
                        break
            elif hasattr(self.category_model, 'feature_importances_'):
                # Direct model with feature_importances_
                if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                    feature_names = self.feature_extractor.get_feature_names_out()
                    for i, importance in enumerate(self.category_model.feature_importances_):
                        if i < len(feature_names):
                            feature_name = feature_names[i]
                            feature_importance[feature_name] = float(importance)

        # Normalize feature importance
        if feature_importance:
            total = sum(feature_importance.values())
            if total > 0:
                feature_importance = {k: v / total for k, v in feature_importance.items()}

        # Sort by importance (descending)
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])

        # Replace feature names with actual words if available
        if feature_descriptions:
            feature_importance_with_descriptions = {}
            for feature, importance in feature_importance.items():
                if feature in feature_descriptions:
                    # Use the actual word as the feature name
                    word = feature_descriptions[feature]
                    feature_importance_with_descriptions[word] = importance
                else:
                    # If no description available, extract the most meaningful part
                    # of the feature name (usually after the last underscore)
                    if '_' in feature:
                        word = feature.split('_')[-1]
                        feature_importance_with_descriptions[word] = importance
                    else:
                        feature_importance_with_descriptions[feature] = importance
            return feature_importance_with_descriptions

        return feature_importance