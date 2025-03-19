import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from collections import defaultdict
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseCategoryClassifier:
    def __init__(self):
        self.category_model = None
        self.subcategory_models = {}  # One model per category
        self.label_encoders = {
            'category': LabelEncoder(),
            'merchant': LabelEncoder()
        }
        self.subcategory_encoders = {}  # One encoder per category
        self.description_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model_path = os.path.join(os.path.dirname(__file__), 'trained_models')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Store category-subcategory mappings
        self.category_subcategories = defaultdict(list)
        self.unique_categories = None

    def preprocess_data(self, df, training=True):
        """
        Preprocess the transaction data.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            training (bool): Whether this is for training or prediction
        """
        # Convert description and merchant to string and clean them
        df['description'] = df['description'].astype(str).str.lower()
        df['merchant'] = df['merchant'].astype(str).str.lower()
        
        # Transform description text to TF-IDF features
        if training:
            description_features = self.description_vectorizer.fit_transform(df['description'])
            # Build category-subcategory mapping
            for category in df['category'].unique():
                subcategories = df[df['category'] == category]['subcategory'].unique()
                self.category_subcategories[category] = list(subcategories)
        else:
            description_features = self.description_vectorizer.transform(df['description'])
        
        # Encode merchant
        if training:
            merchant_encoded = self.label_encoders['merchant'].fit_transform(df['merchant'])
            self.unique_categories = sorted(df['category'].unique())
        else:
            # Handle unknown merchants during prediction
            unknown_merchant = ~df['merchant'].isin(self.label_encoders['merchant'].classes_)
            df.loc[unknown_merchant, 'merchant'] = 'unknown'
            merchant_encoded = self.label_encoders['merchant'].transform(df['merchant'])
        
        # Combine features
        X = np.hstack([
            description_features.toarray(),
            merchant_encoded.reshape(-1, 1)
        ])
        
        # Encode category
        y_category = None
        if training:
            y_category = self.label_encoders['category'].fit_transform(df['category'])
            
            # Create subcategory encoders for each category
            for category in df['category'].unique():
                subcategories = df[df['category'] == category]['subcategory'].unique()
                encoder = LabelEncoder()
                encoder.fit(subcategories)
                self.subcategory_encoders[category] = encoder
        
        return X, y_category, df if training else None

    def train_model(self, training_data, test_size=0.2, random_state=42):
        """Train the Random Forest classifiers using a hierarchical approach."""
        try:
            logger.info("Starting model training...")
            
            # Convert training data to DataFrame if it's not already
            if not isinstance(training_data, pd.DataFrame):
                training_data = pd.DataFrame(training_data)

            # Preprocess data
            logger.info("Preprocessing training data...")
            X, y_category, df = self.preprocess_data(training_data, training=True)
            
            # Split data for category model
            logger.info("Splitting data into train and test sets...")
            try:
                X_train, X_test, y_cat_train, y_cat_test, df_train, df_test = train_test_split(
                    X, y_category, training_data,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y_category
                )
            except ValueError as e:
                logger.warning(f"Stratified split failed: {str(e)}. Falling back to random split.")
                X_train, X_test, y_cat_train, y_cat_test, df_train, df_test = train_test_split(
                    X, y_category, training_data,
                    test_size=test_size,
                    random_state=random_state
                )

            # Train category model
            logger.info("Training category model...")
            self.category_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight='balanced'
            )
            self.category_model.fit(X_train, y_cat_train)
            
            # Train subcategory models for each category
            logger.info("Training subcategory models...")
            subcategory_accuracies = {}
            
            for category in self.unique_categories:
                # Get data for this category
                category_mask_train = df_train['category'] == category
                category_mask_test = df_test['category'] == category
                
                if not any(category_mask_train) or not any(category_mask_test):
                    continue
                
                X_cat_train = X_train[category_mask_train]
                X_cat_test = X_test[category_mask_test]
                
                y_subcat_train = self.subcategory_encoders[category].transform(
                    df_train[category_mask_train]['subcategory']
                )
                y_subcat_test = self.subcategory_encoders[category].transform(
                    df_test[category_mask_test]['subcategory']
                )
                
                # Train subcategory model for this category
                subcat_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=random_state,
                    class_weight='balanced'
                )
                subcat_model.fit(X_cat_train, y_subcat_train)
                self.subcategory_models[category] = subcat_model
                
                # Calculate accuracy for this category's subcategories
                if len(X_cat_test) > 0:
                    y_subcat_pred = subcat_model.predict(X_cat_test)
                    accuracy = np.mean(y_subcat_pred == y_subcat_test)
                    subcategory_accuracies[category] = accuracy
            
            # Evaluate category model
            y_cat_pred = self.category_model.predict(X_test)
            cat_accuracy = np.mean(y_cat_pred == y_cat_test)
            
            # Get feature importance information
            description_features = self.description_vectorizer.get_feature_names_out()
            all_features = list(description_features) + ['merchant']
            
            # Calculate feature importance for category model
            cat_importance = dict(zip(all_features, self.category_model.feature_importances_))
            
            # Save the models and preprocessing objects
            logger.info("Saving models and preprocessing objects...")
            self.save_model()
            
            # Log statistics
            logger.info(f"Number of unique categories: {len(self.unique_categories)}")
            logger.info(f"Category model accuracy: {cat_accuracy:.4f}")
            logger.info("\nSubcategory accuracies by category:")
            for category, accuracy in subcategory_accuracies.items():
                logger.info(f"{category}: {accuracy:.4f}")
            
            return {
                'category_accuracy': float(cat_accuracy),
                'subcategory_accuracies': subcategory_accuracies,
                'category_feature_importance': cat_importance,
                'category_subcategory_mapping': dict(self.category_subcategories),
                'unique_categories': self.unique_categories
            }
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def save_model(self):
        """Save the trained models and preprocessing objects."""
        objects_to_save = {
            'category_model': self.category_model,
            'subcategory_models': self.subcategory_models,
            'label_encoders': self.label_encoders,
            'subcategory_encoders': self.subcategory_encoders,
            'description_vectorizer': self.description_vectorizer,
            'category_subcategories': dict(self.category_subcategories),
            'unique_categories': self.unique_categories
        }
        
        joblib.dump(objects_to_save, os.path.join(self.model_path, 'expense_classifier.joblib'))

    def load_model(self):
        """Load the trained models and preprocessing objects."""
        model_file = os.path.join(self.model_path, 'expense_classifier.joblib')
        
        if os.path.exists(model_file):
            objects = joblib.load(model_file)
            self.category_model = objects['category_model']
            self.subcategory_models = objects['subcategory_models']
            self.label_encoders = objects['label_encoders']
            self.subcategory_encoders = objects['subcategory_encoders']
            self.description_vectorizer = objects['description_vectorizer']
            self.category_subcategories = defaultdict(list, objects['category_subcategories'])
            self.unique_categories = objects['unique_categories']
            return True
        return False

    def predict(self, transaction_data):
        """
        Predict both category and subcategory for a transaction using the hierarchical approach.
        
        Args:
            transaction_data (dict): Dictionary containing transaction information
                                   (description, merchant)
        
        Returns:
            dict: Predicted category and subcategory with confidence scores
        """
        if not isinstance(transaction_data, pd.DataFrame):
            transaction_data = pd.DataFrame([transaction_data])
        
        # Preprocess the input data
        X, _, _ = self.preprocess_data(transaction_data, training=False)
        
        # Predict category
        cat_pred = self.category_model.predict(X)
        cat_prob = self.category_model.predict_proba(X)
        category = self.label_encoders['category'].inverse_transform(cat_pred)[0]
        category_confidence = float(max(cat_prob[0]))
        
        # Predict subcategory using the appropriate model for the predicted category
        if category in self.subcategory_models:
            subcat_model = self.subcategory_models[category]
            subcat_pred = subcat_model.predict(X)
            subcat_prob = subcat_model.predict_proba(X)
            subcategory = self.subcategory_encoders[category].inverse_transform(subcat_pred)[0]
            subcategory_confidence = float(max(subcat_prob[0]))
        else:
            # Fallback if we don't have a model for this category
            subcategory = "Unknown"
            subcategory_confidence = 0.0
        
        return {
            'category': category,
            'category_confidence': category_confidence,
            'subcategory': subcategory,
            'subcategory_confidence': subcategory_confidence,
            'possible_subcategories': self.category_subcategories[category]
        }

# Function to train model with the provided transaction data
def train_with_transaction_data(transactions_data):
    """Train the model with the provided transaction data."""
    # Convert the data to a DataFrame
    df = pd.DataFrame(transactions_data)
    
    # Initialize and train the classifier
    classifier = ExpenseCategoryClassifier()
    results = classifier.train_model(df)
    
    return results 