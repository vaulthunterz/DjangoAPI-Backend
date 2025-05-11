"""
Enhanced feature extraction for expense categorization model
Extracts key terms, entities, and patterns from descriptions and merchants
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

from ..config import FEATURE_WEIGHTS, MAX_FEATURES, NGRAM_RANGE

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract enhanced features from text fields"""

    def __init__(self, text_columns=None, max_features=None, ngram_range=None, column_weights=None):
        self.text_columns = text_columns or ['description', 'merchant']
        self.max_features = max_features or MAX_FEATURES
        self.ngram_range = ngram_range or NGRAM_RANGE
        # Use weights from config if not provided
        self.column_weights = column_weights or FEATURE_WEIGHTS.copy()
        self.vectorizers = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.feature_names = []
        self.feature_descriptions = {}  # Maps feature names to human-readable descriptions

    def fit(self, X, y=None):
        """Fit TF-IDF vectorizers for each text column"""
        for column in self.text_columns:
            # Skip columns with zero weight
            if self.column_weights.get(column, 1.0) == 0.0:
                continue

            if column in X.columns:
                # Create and fit vectorizer for this column
                vectorizer = TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    stop_words='english',
                    preprocessor=self._preprocess_text
                )
                vectorizer.fit(X[column].fillna(''))
                self.vectorizers[column] = vectorizer
        return self

    def transform(self, X):
        """Transform text columns into TF-IDF features with column-specific weights"""
        result = pd.DataFrame(index=X.index)
        self.feature_names = []
        self.feature_descriptions = {}

        # Transform each text column
        for column, vectorizer in self.vectorizers.items():
            # Skip columns with zero weight
            weight = self.column_weights.get(column, 1.0)
            if weight == 0.0:
                continue

            if column in X.columns:
                # Transform text to TF-IDF features
                tfidf_features = vectorizer.transform(X[column].fillna(''))

                # Get raw feature names from vectorizer
                raw_feature_names = vectorizer.get_feature_names_out()

                # Create descriptive feature names
                feature_names = [f"{column}_{name}" for name in raw_feature_names]

                # Apply weight to the features
                weighted_features = tfidf_features.multiply(weight)

                # Store the actual words as feature descriptions
                for i, name in enumerate(feature_names):
                    raw_name = raw_feature_names[i]
                    # Store the actual word as the description
                    self.feature_descriptions[name] = raw_name

                # Create DataFrame with feature names
                tfidf_df = pd.DataFrame(
                    weighted_features.toarray(),
                    columns=feature_names,
                    index=X.index
                )

                # Add to result and track feature names
                result = pd.concat([result, tfidf_df], axis=1)
                self.feature_names.extend(feature_names)

        # Add custom extracted features
        custom_features = self._extract_custom_features(X)
        result = pd.concat([result, custom_features], axis=1)

        # Add custom feature names and descriptions
        self.feature_names.extend(custom_features.columns)

        # Add numeric columns from original data
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            result = pd.concat([result, X[numeric_cols]], axis=1)
            self.feature_names.extend(numeric_cols)

            # Add descriptions for numeric columns
            for col in numeric_cols:
                self.feature_descriptions[col] = f"Numeric value of {col}"

        return result

    def _preprocess_text(self, text):
        """Preprocess text by removing punctuation, lowercasing, and lemmatizing"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(f'[{string.punctuation}]', ' ', text)

        # Simple tokenization to avoid NLTK issues
        tokens = text.split()

        # Remove stopwords and lemmatize
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        return ' '.join(lemmatized)

    def _extract_custom_features(self, X):
        """Extract custom features from text fields"""
        result = pd.DataFrame(index=X.index)

        # Dictionary to store feature descriptions
        descriptions = {}

        # Only process if we have the necessary columns
        if 'description' in X.columns:
            desc = X['description'].fillna('').astype(str)

            # Feature: Description length
            result['desc_length'] = desc.apply(len)
            descriptions['desc_length'] = "length"

            # Feature: Word count in description
            result['desc_word_count'] = desc.apply(lambda x: len(x.split()))
            descriptions['desc_word_count'] = "wordcount"

            # Feature: Contains numbers
            result['desc_has_numbers'] = desc.apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)
            descriptions['desc_has_numbers'] = "numbers"

            # Feature: Contains currency symbols
            result['desc_has_currency'] = desc.apply(lambda x: 1 if any(c in '$£€¥' for c in x) else 0)
            descriptions['desc_has_currency'] = "currency"

            # Feature: Contains specific keywords
            keywords = {
                'payment': 'keyword_payment',
                'bill': 'keyword_bill',
                'subscription': 'keyword_subscription',
                'food': 'keyword_food',
                'transport': 'keyword_transport',
                'health': 'keyword_health',
                'travel': 'keyword_travel',
                'shopping': 'keyword_shopping'
            }

            # Use the actual keyword as the feature description
            for keyword, feature_name in keywords.items():
                result[feature_name] = desc.apply(lambda x: 1 if keyword.lower() in x.lower() else 0)
                descriptions[feature_name] = keyword

        # Extract features from merchant if available
        if 'merchant' in X.columns:
            merchant = X['merchant'].fillna('').astype(str)

            # Feature: Merchant name length
            result['merchant_length'] = merchant.apply(len)
            descriptions['merchant_length'] = "merchant_length"

            # Feature: Merchant contains specific business types
            business_types = {
                'restaurant': 'merchant_restaurant',
                'hotel': 'merchant_hotel',
                'hospital': 'merchant_hospital',
                'shop': 'merchant_shop',
                'market': 'merchant_market',
                'bank': 'merchant_bank',
                'university': 'merchant_university',
                'school': 'merchant_school'
            }

            # Use the actual business type as the feature description
            for business_type, feature_name in business_types.items():
                result[feature_name] = merchant.apply(lambda x: 1 if business_type.lower() in x.lower() else 0)
                descriptions[feature_name] = business_type

        # Update feature descriptions
        self.feature_descriptions.update(descriptions)

        return result

    def get_feature_names_out(self):
        """Get feature names"""
        return self.feature_names

    def get_feature_descriptions(self):
        """Get feature descriptions dictionary"""
        return self.feature_descriptions

    def get_readable_feature_names(self):
        """Get a dictionary mapping feature names to human-readable descriptions"""
        return {name: self.feature_descriptions.get(name, name) for name in self.feature_names}


# Function to create enhanced features for model training
def create_enhanced_features(df, feature_weights=None):
    """
    Create enhanced features from the input DataFrame

    Args:
        df: Input DataFrame with transaction data
        feature_weights: Dictionary of weights to apply to different features
                        (default: use weights from config)
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()

    # Fill missing values in text columns
    for col in ['description', 'merchant']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('').astype(str)

    # Extract amount as a numeric feature if it exists
    if 'amount' in df_copy.columns:
        df_copy['amount'] = pd.to_numeric(df_copy['amount'], errors='coerce')
        # Fill missing amounts with the median
        median_amount = df_copy['amount'].median()
        df_copy['amount'] = df_copy['amount'].fillna(median_amount)

    # Apply the text feature extractor with weights from config or provided weights
    feature_extractor = TextFeatureExtractor(
        text_columns=['description', 'merchant'],
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        column_weights=feature_weights
    )

    # Fit and transform
    features = feature_extractor.fit_transform(df_copy)

    # Replace any remaining NaN values with zeros
    features = features.fillna(0)

    return features, feature_extractor