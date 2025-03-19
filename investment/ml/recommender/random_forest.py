"""
RandomForest-based recommender implementation
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import List, Dict, Any

from .base import BaseRecommender
from django.conf import settings

class RandomForestRecommender(BaseRecommender):
    def __init__(self):
        self.model_dir = os.path.join(settings.BASE_DIR, 'ml', 'models')
        self.model_path = os.path.join(self.model_dir, 'rf_recommender.joblib')
        
        self.feature_columns = [
            'risk_tolerance', 
            'investment_experience', 
            'investment_timeline',
            'monthly_disposable_income', 
            'age_group'
        ]
        
        self.numeric_features = ['monthly_disposable_income']
        self.categorical_features = [
            'risk_tolerance', 
            'investment_experience', 
            'investment_timeline', 
            'age_group'
        ]
        
        self.model = self._create_pipeline()
        
    def _create_pipeline(self) -> Pipeline:
        """Create the ML pipeline with preprocessing and model"""
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        return Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ))
        ])

    def train(self, training_data: pd.DataFrame) -> None:
        """Train the model with historical data"""
        if training_data is not None and not training_data.empty:
            X = training_data[self.feature_columns]
            y = training_data['target']
            self.model.fit(X, y)
            self.save_model(self.model_path)

    def predict(self, user_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on user features"""
        features_df = pd.DataFrame([user_features])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = None
        
        # Make prediction
        probabilities = self.model.predict_proba(features_df)
        
        # Get class labels
        classes = self.model.classes_
        
        # Create recommendations
        recommendations = []
        for i, class_label in enumerate(classes):
            recommendations.append({
                'asset_type': class_label,
                'probability': float(probabilities[0][i]),
                'confidence': 'high' if probabilities[0][i] > 0.7 else 'medium' if probabilities[0][i] > 0.4 else 'low'
            })
        
        # Sort by probability
        recommendations.sort(key=lambda x: x['probability'], reverse=True)
        return recommendations

    def update(self, feedback_data: pd.DataFrame) -> None:
        """Update the model with new feedback data"""
        if feedback_data is not None and not feedback_data.empty:
            # Combine new data with existing if needed
            self.train(feedback_data)

    def save_model(self, path: str) -> None:
        """Save the model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load_model(self, path: str) -> None:
        """Load the model from disk"""
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            self.model = self._create_pipeline() 