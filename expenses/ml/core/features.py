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

    def __init__(self, text_columns=None, max_features=1000, ngram_range=(1, 2)):
        self.text_columns = text_columns or ['description', 'merchant']
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizers = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        """Fit TF-IDF vectorizers for each text column"""
        for column in self.text_columns:
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
        """Transform text columns into TF-IDF features"""
        result = pd.DataFrame(index=X.index)

        # Transform each text column
        for column, vectorizer in self.vectorizers.items():
            if column in X.columns:
                # Transform text to TF-IDF features
                tfidf_features = vectorizer.transform(X[column].fillna(''))

                # Convert to DataFrame with feature names
                feature_names = [f"{column}_{name}" for name in vectorizer.get_feature_names_out()]
                tfidf_df = pd.DataFrame(
                    tfidf_features.toarray(),
                    columns=feature_names,
                    index=X.index
                )

                # Add to result
                result = pd.concat([result, tfidf_df], axis=1)

        # Add custom extracted features
        result = pd.concat([result, self._extract_custom_features(X)], axis=1)

        # Add numeric columns from original data
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            result = pd.concat([result, X[numeric_cols]], axis=1)

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

        # Only process if we have the necessary columns
        if 'description' in X.columns:
            desc = X['description'].fillna('').astype(str)

            # Feature: Description length
            result['desc_length'] = desc.apply(len)

            # Feature: Word count in description
            result['desc_word_count'] = desc.apply(lambda x: len(x.split()))

            # Feature: Contains numbers
            result['desc_has_numbers'] = desc.apply(lambda x: 1 if any(c.isdigit() for c in x) else 0)

            # Feature: Contains currency symbols
            result['desc_has_currency'] = desc.apply(lambda x: 1 if any(c in '$£€¥' for c in x) else 0)

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

            for keyword, feature_name in keywords.items():
                result[feature_name] = desc.apply(lambda x: 1 if keyword.lower() in x.lower() else 0)

        # Extract features from merchant if available
        if 'merchant' in X.columns:
            merchant = X['merchant'].fillna('').astype(str)

            # Feature: Merchant name length
            result['merchant_length'] = merchant.apply(len)

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

            for business_type, feature_name in business_types.items():
                result[feature_name] = merchant.apply(lambda x: 1 if business_type.lower() in x.lower() else 0)

        return result

# Function to create enhanced features for model training
def create_enhanced_features(df):
    """Create enhanced features from the input DataFrame"""
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

    # Apply the text feature extractor
    feature_extractor = TextFeatureExtractor(
        text_columns=['description', 'merchant'],
        max_features=500,
        ngram_range=(1, 3)
    )

    # Fit and transform
    features = feature_extractor.fit_transform(df_copy)

    # Replace any remaining NaN values with zeros
    features = features.fillna(0)

    return features, feature_extractor