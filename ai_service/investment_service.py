"""
Investment AI Service Module

This module provides AI functionality for investment recommendations and portfolio analysis.
"""
import os
import logging
import pandas as pd
import joblib
from typing import Dict, Any, List, Optional, Union

from .service import AIService

# Import utility functions to avoid circular imports
def _import_investment_ml():
    """Import investment ML modules lazily to avoid circular imports."""
    from investment.ml.recommender.advanced_hybrid_recommender import AdvancedHybridRecommender
    from investment.ml.recommender.hybrid_recommender import HybridRecommender
    from investment.ml.recommender.random_forest import RandomForestRecommender
    from investment.ml.utils.preprocessing import prepare_user_features, calculate_portfolio_metrics

    return (
        AdvancedHybridRecommender,
        HybridRecommender,
        RandomForestRecommender,
        prepare_user_features,
        calculate_portfolio_metrics
    )

# Configure logging
logger = logging.getLogger(__name__)

class InvestmentAIService(AIService):
    """
    AI service for investment recommendations and portfolio analysis.

    This service provides functionality for:
    - Generating personalized investment recommendations
    - Analyzing portfolio performance
    - Training recommendation models
    """

    def __init__(self):
        """Initialize the Investment AI service."""
        super().__init__()
        self.advanced_recommender = None
        self.hybrid_recommender = None
        self.random_forest_recommender = None

    def initialize(self) -> bool:
        """
        Initialize the Investment AI service and load required models.

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Import ML modules lazily to avoid circular imports
            AdvancedHybridRecommender, HybridRecommender, RandomForestRecommender, \
            prepare_user_features, calculate_portfolio_metrics = _import_investment_ml()

            # Store the imported functions for later use
            self._prepare_user_features = prepare_user_features
            self._calculate_portfolio_metrics = calculate_portfolio_metrics

            # Initialize recommenders
            self.advanced_recommender = AdvancedHybridRecommender()
            self.hybrid_recommender = HybridRecommender()
            self.random_forest_recommender = RandomForestRecommender()

            self.models = {
                "advanced_recommender": True,
                "hybrid_recommender": True,
                "random_forest_recommender": True
            }

            self.initialized = True
            logger.info("Investment AI service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing Investment AI service: {str(e)}")
            self.initialized = False
            return False

    def predict(self, data: Dict[str, Any], recommender_type: str = 'advanced') -> List[Dict[str, Any]]:
        """
        Generate investment recommendations based on user features.

        Args:
            data: Dictionary containing user profile and questionnaire information
            recommender_type: Type of recommender to use ('advanced', 'hybrid', or 'random_forest')

        Returns:
            List[Dict[str, Any]]: List of investment recommendations
        """
        self._ensure_initialized()

        # Validate input data
        if 'user_profile' not in data:
            raise ValueError("User profile is required for investment recommendations")

        # Select the appropriate recommender
        recommender = self._get_recommender(recommender_type)

        # Generate recommendations
        try:
            recommendations = recommender.predict(data)
            return recommendations
        except Exception as e:
            logger.error(f"Error generating investment recommendations: {str(e)}")
            raise

    def get_expense_based_recommendations(self, user_profile: Any) -> List[Dict[str, Any]]:
        """
        Generate investment recommendations based on user's expense patterns.

        Args:
            user_profile: User profile object

        Returns:
            List[Dict[str, Any]]: List of expense-based investment recommendations
        """
        self._ensure_initialized()

        # Prepare user features
        user_features = {
            'user_profile': user_profile,
            'questionnaire': None
        }

        # Get recommendations using the advanced recommender
        try:
            recommendations = self.advanced_recommender.predict(user_features)

            # Filter to only include expense-based recommendations
            expense_based_recs = [rec for rec in recommendations
                                 if rec.get('recommendation_type') == 'expense_based']

            return expense_based_recs
        except Exception as e:
            logger.error(f"Error generating expense-based recommendations: {str(e)}")
            raise

    def analyze_portfolio(self, portfolio: Any) -> Dict[str, Any]:
        """
        Analyze a user's investment portfolio.

        Args:
            portfolio: Portfolio object

        Returns:
            Dict[str, Any]: Portfolio analysis results
        """
        self._ensure_initialized()

        try:
            # Calculate portfolio metrics using the stored function
            metrics = self._calculate_portfolio_metrics(portfolio)

            # Add additional analysis
            metrics['risk_assessment'] = self._assess_portfolio_risk(portfolio)
            metrics['diversification_score'] = self._calculate_diversification_score(portfolio)
            metrics['improvement_suggestions'] = self._generate_improvement_suggestions(portfolio, metrics)

            return metrics
        except Exception as e:
            logger.error(f"Error analyzing portfolio: {str(e)}")
            raise

    def train(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train the investment recommendation models with the provided data.

        Args:
            data: DataFrame containing training data

        Returns:
            Dict[str, Any]: Training results
        """
        self._ensure_initialized()

        try:
            # If no data provided, create synthetic training data
            if data is None:
                from investment.ml.utils.preprocessing import create_synthetic_training_data
                data = create_synthetic_training_data()

            # Train each recommender
            results = {}

            # Train advanced recommender
            self.advanced_recommender.train(data)
            results['advanced_recommender'] = {'status': 'success'}

            # Train hybrid recommender
            self.hybrid_recommender.train(data)
            results['hybrid_recommender'] = {'status': 'success'}

            # Train random forest recommender
            self.random_forest_recommender.train(data)
            results['random_forest_recommender'] = {'status': 'success'}

            logger.info("Investment recommendation models trained successfully")
            return results

        except Exception as e:
            logger.error(f"Error training investment models: {str(e)}")
            raise

    def train_with_user_profiles(self, user_profiles: List[Any], portfolio_performances: List[float]) -> Dict[str, Any]:
        """
        Train the recommendation models with user profiles and portfolio performances.

        Args:
            user_profiles: List of user profile objects
            portfolio_performances: List of portfolio performance values

        Returns:
            Dict[str, Any]: Training results
        """
        self._ensure_initialized()

        try:
            # Prepare training data
            from investment.ml.utils.preprocessing import prepare_training_data
            training_data = prepare_training_data(user_profiles, portfolio_performances)

            # Train the models
            return self.train(training_data)

        except Exception as e:
            logger.error(f"Error training with user profiles: {str(e)}")
            raise

    def _get_recommender(self, recommender_type: str):
        """
        Get the appropriate recommender based on the specified type.

        Args:
            recommender_type: Type of recommender ('advanced', 'hybrid', or 'random_forest')

        Returns:
            Recommender object
        """
        if recommender_type == 'advanced':
            return self.advanced_recommender
        elif recommender_type == 'hybrid':
            return self.hybrid_recommender
        elif recommender_type == 'random_forest':
            return self.random_forest_recommender
        else:
            raise ValueError(f"Unknown recommender type: {recommender_type}")

    def _assess_portfolio_risk(self, portfolio: Any) -> Dict[str, Any]:
        """
        Assess the risk level of a portfolio.

        Args:
            portfolio: Portfolio object

        Returns:
            Dict[str, Any]: Risk assessment results
        """
        # Get portfolio items
        portfolio_items = portfolio.portfolioitem_set.all()

        if not portfolio_items.exists():
            return {
                'risk_level': 'unknown',
                'risk_score': 0,
                'description': 'No investments in portfolio'
            }

        # Calculate risk metrics
        # This is a simplified implementation
        risk_scores = {
            'Stocks': 7,
            'Bonds': 4,
            'ETF': 5,
            'Mutual Fund': 5,
            'Cash': 1,
            'Real Estate': 6,
            'Cryptocurrency': 9,
            'Commodities': 8
        }

        total_value = sum(item.quantity * item.buy_price for item in portfolio_items)
        weighted_risk = 0

        for item in portfolio_items:
            asset_type = item.asset_type if hasattr(item, 'asset_type') else 'Unknown'
            item_value = item.quantity * item.buy_price
            item_weight = item_value / total_value if total_value > 0 else 0
            item_risk = risk_scores.get(asset_type, 5)  # Default risk score of 5 for unknown types
            weighted_risk += item_weight * item_risk

        # Determine risk level
        risk_level = 'low' if weighted_risk < 3 else 'medium' if weighted_risk < 6 else 'high'

        return {
            'risk_level': risk_level,
            'risk_score': round(weighted_risk, 2),
            'description': f'Portfolio has a {risk_level} risk level with a score of {round(weighted_risk, 2)}/10'
        }

    def _calculate_diversification_score(self, portfolio: Any) -> float:
        """
        Calculate a diversification score for a portfolio.

        Args:
            portfolio: Portfolio object

        Returns:
            float: Diversification score (0-10)
        """
        # Get portfolio items
        portfolio_items = portfolio.portfolioitem_set.all()

        if not portfolio_items.exists():
            return 0

        # Count unique asset types
        asset_types = set()
        for item in portfolio_items:
            asset_type = item.asset_type if hasattr(item, 'asset_type') else 'Unknown'
            asset_types.add(asset_type)

        # Calculate concentration
        total_value = sum(item.quantity * item.buy_price for item in portfolio_items)
        type_values = {}

        for item in portfolio_items:
            asset_type = item.asset_type if hasattr(item, 'asset_type') else 'Unknown'
            item_value = item.quantity * item.buy_price

            if asset_type not in type_values:
                type_values[asset_type] = 0
            type_values[asset_type] += item_value

        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        hhi = sum((value / total_value) ** 2 for value in type_values.values()) if total_value > 0 else 1

        # Convert HHI to diversification score (0-10)
        # HHI ranges from 1/n (perfectly diversified) to 1 (completely concentrated)
        # Lower HHI means better diversification
        diversification_score = 10 * (1 - hhi)

        return round(diversification_score, 2)

    def _generate_improvement_suggestions(self, portfolio: Any, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate suggestions for improving a portfolio.

        Args:
            portfolio: Portfolio object
            metrics: Portfolio metrics

        Returns:
            List[str]: Improvement suggestions
        """
        suggestions = []

        # Risk-based suggestions
        risk_assessment = metrics.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'unknown')
        risk_score = risk_assessment.get('risk_score', 0)

        if risk_level == 'high' and risk_score > 7:
            suggestions.append("Consider adding more conservative investments to reduce portfolio risk")
        elif risk_level == 'low' and risk_score < 3:
            suggestions.append("Consider adding growth investments to potentially increase returns")

        # Diversification-based suggestions
        diversification_score = metrics.get('diversification_score', 0)

        if diversification_score < 5:
            suggestions.append("Improve diversification by adding different asset types")

        # Performance-based suggestions
        returns = metrics.get('returns_percentage', 0)

        if returns < 0:
            suggestions.append("Review underperforming investments and consider rebalancing")

        # Add general suggestions if list is empty
        if not suggestions:
            suggestions.append("Regularly review and rebalance your portfolio to maintain optimal allocation")

        return suggestions
