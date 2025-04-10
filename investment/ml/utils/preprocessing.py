"""
Utility functions for data preprocessing
"""
import pandas as pd
from typing import Dict, Any
from ...models import UserProfile, Portfolio, PortfolioItem

def prepare_user_features(user_profile: UserProfile) -> Dict[str, Any]:
    """Convert UserProfile to feature dictionary"""
    return {
        'risk_tolerance': user_profile.risk_tolerance,
        'investment_experience': user_profile.investment_experience,
        'investment_timeline': user_profile.investment_timeline,
        'monthly_disposable_income': float(user_profile.monthly_disposable_income or 0),
        'age_group': 'adult'  # TODO: Calculate based on user's date of birth
    }

def calculate_portfolio_metrics(portfolio: Portfolio) -> Dict[str, float]:
    """Calculate portfolio performance metrics"""
    items = PortfolioItem.objects.filter(portfolio=portfolio)

    total_value = sum(item.quantity * item.buy_price for item in items)
    asset_distribution = {
        item.asset_name: (item.quantity * item.buy_price) / total_value
        for item in items
    } if total_value > 0 else {}

    return {
        'total_value': total_value,
        'asset_distribution': asset_distribution,
        # Add more metrics as needed:
        # 'returns': calculate_returns(),
        # 'volatility': calculate_volatility(),
        # 'sharpe_ratio': calculate_sharpe_ratio(),
    }

def prepare_training_data(user_profiles: list, portfolio_performances: list) -> pd.DataFrame:
    """Prepare training data from user profiles and portfolio performances"""
    training_data = []

    for profile, performance in zip(user_profiles, portfolio_performances):
        features = prepare_user_features(profile)
        # Add performance metrics as target
        features['target'] = 'successful' if performance > 0.1 else 'moderate' if performance > 0 else 'poor'
        training_data.append(features)

    return pd.DataFrame(training_data)