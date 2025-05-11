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

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.category_model = None
        self.subcategory_models = {}
        self.feature_extractor = None
        self.categories = None
        self.subcategories = {}

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

    def train(self, data_path, test_size=0.2, random_state=42):
        """Train the ensemble models"""
        # Start timing
        start_time = time.time()

        # Print header for progress tracking
        print("\n" + "="*70)
        print(" ENHANCED EXPENSE CATEGORIZATION MODEL TRAINING")
        print("="*70)

        # Step 1: Load data
        print("\n[1/5] Loading training data...")
        df = pd.read_csv(data_path)
        print(f"      Loaded {len(df)} transactions with {df['category'].nunique()} categories")

        # Step 2: Create enhanced features
        print("\n[2/5] Creating enhanced features...")
        feature_start = time.time()
        sys.stdout.write("      Extracting features... ")
        sys.stdout.flush()
        X, self.feature_extractor = create_enhanced_features(df)

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
        for model in self.category_model.models:
            if isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier):
                model.verbose = 1
            elif isinstance(model, Pipeline):
                # For Pipeline objects, we need to set verbose on the classifier
                if hasattr(model, 'steps'):
                    for name, step in model.steps:
                        if hasattr(step, 'verbose'):
                            step.verbose = 1

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

            # Extract features for this category
            X_cat, _ = create_enhanced_features(category_df)

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

        # Save models and metadata
        print("\n      Saving models and metadata...")
        self._save_models()

        # Calculate total training time
        total_time = time.time() - start_time

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
                # Fallback to creating new features
                X, _ = create_enhanced_features(transaction)

            # Predict category with confidence scores
            category_probs = self.category_model.predict_proba(X)[0]
            category_idx = np.argmax(category_probs)
            category = self.category_model.classes_[category_idx]
            category_confidence = float(category_probs[category_idx])

            # Get top 3 category predictions with confidence scores
            top_categories = []
            sorted_indices = np.argsort(category_probs)[::-1][:3]  # Top 3 indices
            for idx in sorted_indices:
                if idx < len(self.category_model.classes_):
                    top_categories.append({
                        'category': self.category_model.classes_[idx],
                        'confidence': float(category_probs[idx])
                    })

            # Predict subcategory if we have a model for this category
            subcategory = None
            subcategory_confidence = 0.0
            top_subcategories = []

            if category in self.subcategory_models:
                subcategory_model = self.subcategory_models[category]
                subcategory_probs = subcategory_model.predict_proba(X)[0]
                subcategory_idx = np.argmax(subcategory_probs)
                subcategory = subcategory_model.classes_[subcategory_idx]
                subcategory_confidence = float(subcategory_probs[subcategory_idx])

                # Get top 3 subcategory predictions with confidence scores
                sorted_indices = np.argsort(subcategory_probs)[::-1][:3]  # Top 3 indices
                for idx in sorted_indices:
                    if idx < len(subcategory_model.classes_):
                        top_subcategories.append({
                            'subcategory': subcategory_model.classes_[idx],
                            'confidence': float(subcategory_probs[idx])
                        })

            # Get individual model predictions for the category
            individual_model_predictions = []
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
        # Create individual models with imputation for missing values
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            ))
        ])

        gb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=1  # Show progress
            ))
        ])

        lr_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=1.0,
                max_iter=1000,
                multi_class='multinomial',
                random_state=42,
                n_jobs=-1  # Use all available cores
            ))
        ])

        # Create ensemble with weights favoring RandomForest and GradientBoosting
        ensemble = EnsembleClassifier(
            models=[rf_pipeline, gb_pipeline, lr_pipeline],
            weights=[0.5, 0.3, 0.2]  # Give more weight to RandomForest
        )

        return ensemble

    def _create_subcategory_ensemble(self):
        """Create ensemble model for subcategory prediction"""
        # Create individual models with imputation for missing values
        rf_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all available cores
            ))
        ])

        gb_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('classifier', GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                verbose=0  # Don't show progress for subcategory models to avoid clutter
            ))
        ])

        # Create ensemble
        ensemble = EnsembleClassifier(
            models=[rf_pipeline, gb_pipeline],
            weights=[0.6, 0.4]  # Give more weight to RandomForest
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

        # Save metadata
        metadata = {
            'categories': self.categories,
            'subcategories': self.subcategories
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
        """Get feature importance from the category model"""
        # Only works for RandomForest and GradientBoosting models
        feature_importance = {}

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
                                    feature_importance[feature_names[i]] = importance
                        break
            elif hasattr(model, 'feature_importances_'):
                # Direct model with feature_importances_
                if self.feature_extractor and hasattr(self.feature_extractor, 'get_feature_names_out'):
                    feature_names = self.feature_extractor.get_feature_names_out()
                    for i, importance in enumerate(model.feature_importances_):
                        if i < len(feature_names):
                            feature_importance[feature_names[i]] = importance

        return feature_importance