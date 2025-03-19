import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from typing import List, Dict, Any
from .models import UserProfile, FinancialAsset, Portfolio
from django.conf import settings
import os

class MLRecommender:
    def __init__(self):
        self.model_path = os.path.join(settings.BASE_DIR, 'investment', 'ml_models', 'recommender_model.joblib')
        self.feature_columns = [
            'risk_tolerance', 'investment_experience', 'investment_timeline',
            'monthly_disposable_income', 'age_group'
        ]
        self.model = self._load_or_create_model()

    def _load_or_create_model(self):
        """Load existing model or create a new one if it doesn't exist"""
        try:
            return joblib.load(self.model_path)
        except:
            return self._create_pipeline()

    def _create_pipeline(self):
        """Create the ML pipeline with preprocessing and model"""
        numeric_features = ['monthly_disposable_income']
        categorical_features = ['risk_tolerance', 'investment_experience', 'investment_timeline', 'age_group']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        return pipeline

    def prepare_user_features(self, user_profile: UserProfile) -> pd.DataFrame:
        """Convert user profile to features dataframe"""
        # Calculate age group based on user's date of birth if available
        age_group = 'adult'  # Default value, you can enhance this based on actual user data

        user_data = {
            'risk_tolerance': user_profile.risk_tolerance,
            'investment_experience': user_profile.investment_experience,
            'investment_timeline': user_profile.investment_timeline,
            'monthly_disposable_income': float(user_profile.monthly_disposable_income or 0),
            'age_group': age_group
        }
        
        return pd.DataFrame([user_data])

    def get_asset_recommendations(self, user_profile: UserProfile, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get personalized asset recommendations for a user"""
        # Prepare user features
        user_features = self.prepare_user_features(user_profile)
        
        # Get all available assets
        assets = FinancialAsset.objects.all()
        
        recommendations = []
        for asset in assets:
            # Calculate recommendation score using the model
            # This is a simplified version - you would normally predict probability scores
            features = user_features.copy()
            score = np.random.random()  # Replace with actual model prediction
            
            recommendations.append({
                'asset': asset,
                'score': score,
                'reason': self._generate_recommendation_reason(asset, user_profile)
            })
        
        # Sort by score and get top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]

    def _generate_recommendation_reason(self, asset: FinancialAsset, user_profile: UserProfile) -> str:
        """Generate a personalized reason for recommending an asset"""
        reasons = []
        
        # Risk alignment
        if asset.risk_level <= user_profile.risk_tolerance:
            reasons.append("This asset's risk level aligns with your risk tolerance")
        
        # Investment timeline consideration
        if user_profile.investment_timeline == 'long' and asset.asset_type in ['Stock', 'ETF']:
            reasons.append("Suitable for your long-term investment horizon")
        elif user_profile.investment_timeline == 'short' and asset.asset_type in ['Bond']:
            reasons.append("Matches your short-term investment timeline")
        
        # Experience level consideration
        if user_profile.investment_experience == 'beginner' and asset.asset_type in ['ETF', 'Bond']:
            reasons.append("Appropriate for your investment experience level")
        
        return " and ".join(reasons) if reasons else "Based on your overall investment profile"

    def train_model(self, training_data: pd.DataFrame):
        """Train the model with historical data"""
        if training_data is not None and not training_data.empty:
            X = training_data[self.feature_columns]
            y = training_data['target']  # This would be your target variable
            
            self.model.fit(X, y)
            
            # Save the trained model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)

    def update_recommendations(self, user_profile: UserProfile) -> None:
        """Update recommendations based on user's portfolio performance and market data"""
        # Get user's portfolio
        portfolios = Portfolio.objects.filter(user=user_profile)
        
        # Here you would:
        # 1. Analyze portfolio performance
        # 2. Update user risk profile if needed
        # 3. Generate new recommendations
        # 4. Store recommendations in the database
        
        recommendations = self.get_asset_recommendations(user_profile)
        for rec in recommendations:
            # Store recommendation in your database
            pass 