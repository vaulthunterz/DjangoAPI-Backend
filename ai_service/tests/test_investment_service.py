"""
Tests for the InvestmentAIService class.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from ai_service.investment_service import InvestmentAIService


class TestInvestmentAIService(unittest.TestCase):
    """Test cases for the InvestmentAIService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = InvestmentAIService()
        
        # Sample data for testing
        self.sample_user_profile = {
            'user_profile': {
                'id': 1,
                'age': 30,
                'risk_tolerance': 'moderate',
                'investment_horizon': 'long_term'
            },
            'questionnaire': {
                'goal': 'retirement',
                'income': 75000,
                'savings': 25000
            }
        }
        
        # Create a mock portfolio
        self.mock_portfolio = MagicMock()
        self.mock_portfolio_item1 = MagicMock()
        self.mock_portfolio_item1.asset_type = 'Stocks'
        self.mock_portfolio_item1.quantity = 10
        self.mock_portfolio_item1.buy_price = 100
        
        self.mock_portfolio_item2 = MagicMock()
        self.mock_portfolio_item2.asset_type = 'Bonds'
        self.mock_portfolio_item2.quantity = 5
        self.mock_portfolio_item2.buy_price = 200
        
        # Set up portfolio items
        self.mock_portfolio.portfolioitem_set = MagicMock()
        self.mock_portfolio.portfolioitem_set.all.return_value = [
            self.mock_portfolio_item1,
            self.mock_portfolio_item2
        ]
        self.mock_portfolio.portfolioitem_set.exists.return_value = True
    
    def test_initialize(self):
        """Test initializing the service."""
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(self.service.initialized)
        self.assertIsNotNone(self.service.advanced_recommender)
        self.assertIsNotNone(self.service.hybrid_recommender)
        self.assertIsNotNone(self.service.random_forest_recommender)
        
        # Check that all recommenders are in the models dictionary
        self.assertIn('advanced_recommender', self.service.models)
        self.assertIn('hybrid_recommender', self.service.models)
        self.assertIn('random_forest_recommender', self.service.models)
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    def test_predict(self, mock_ensure_initialized, mock_initialize):
        """Test predicting investment recommendations."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the advanced recommender
        self.service.advanced_recommender = MagicMock()
        self.service.advanced_recommender.predict.return_value = [
            {'asset_type': 'Stocks', 'allocation': 60, 'recommendation_type': 'risk_based'},
            {'asset_type': 'Bonds', 'allocation': 30, 'recommendation_type': 'goal_based'},
            {'asset_type': 'Cash', 'allocation': 10, 'recommendation_type': 'expense_based'}
        ]
        
        # Make a prediction
        result = self.service.predict(self.sample_user_profile)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the advanced recommender's predict method was called
        self.service.advanced_recommender.predict.assert_called_once_with(self.sample_user_profile)
        
        # Check the prediction result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['asset_type'], 'Stocks')
        self.assertEqual(result[1]['asset_type'], 'Bonds')
        self.assertEqual(result[2]['asset_type'], 'Cash')
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    def test_predict_with_different_recommender(self, mock_ensure_initialized, mock_initialize):
        """Test predicting with a different recommender."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the hybrid recommender
        self.service.hybrid_recommender = MagicMock()
        self.service.hybrid_recommender.predict.return_value = [
            {'asset_type': 'Stocks', 'allocation': 50},
            {'asset_type': 'Bonds', 'allocation': 40},
            {'asset_type': 'Cash', 'allocation': 10}
        ]
        
        # Make a prediction with the hybrid recommender
        result = self.service.predict(self.sample_user_profile, recommender_type='hybrid')
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the hybrid recommender's predict method was called
        self.service.hybrid_recommender.predict.assert_called_once_with(self.sample_user_profile)
        
        # Check the prediction result
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['asset_type'], 'Stocks')
        self.assertEqual(result[1]['asset_type'], 'Bonds')
        self.assertEqual(result[2]['asset_type'], 'Cash')
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    def test_get_expense_based_recommendations(self, mock_ensure_initialized, mock_initialize):
        """Test getting expense-based recommendations."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the advanced recommender
        self.service.advanced_recommender = MagicMock()
        self.service.advanced_recommender.predict.return_value = [
            {'asset_type': 'Stocks', 'allocation': 60, 'recommendation_type': 'risk_based'},
            {'asset_type': 'Bonds', 'allocation': 30, 'recommendation_type': 'goal_based'},
            {'asset_type': 'ETF', 'allocation': 10, 'recommendation_type': 'expense_based'}
        ]
        
        # Get expense-based recommendations
        result = self.service.get_expense_based_recommendations(self.sample_user_profile['user_profile'])
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the advanced recommender's predict method was called
        self.service.advanced_recommender.predict.assert_called_once()
        
        # Check the recommendation result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['asset_type'], 'ETF')
        self.assertEqual(result[0]['recommendation_type'], 'expense_based')
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    @patch('investment.ml.utils.preprocessing.calculate_portfolio_metrics')
    def test_analyze_portfolio(self, mock_calculate_metrics, mock_ensure_initialized, mock_initialize):
        """Test analyzing a portfolio."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock calculate_portfolio_metrics
        mock_calculate_metrics.return_value = {
            'total_invested': 2000,
            'current_value': 2200,
            'returns': 200,
            'returns_percentage': 10
        }
        
        # Analyze the portfolio
        result = self.service.analyze_portfolio(self.mock_portfolio)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that calculate_portfolio_metrics was called
        mock_calculate_metrics.assert_called_once_with(self.mock_portfolio)
        
        # Check the analysis result
        self.assertEqual(result['total_invested'], 2000)
        self.assertEqual(result['current_value'], 2200)
        self.assertEqual(result['returns'], 200)
        self.assertEqual(result['returns_percentage'], 10)
        self.assertIn('risk_assessment', result)
        self.assertIn('diversification_score', result)
        self.assertIn('improvement_suggestions', result)
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    def test_train(self, mock_ensure_initialized, mock_initialize):
        """Test training the investment models."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the recommenders
        self.service.advanced_recommender = MagicMock()
        self.service.hybrid_recommender = MagicMock()
        self.service.random_forest_recommender = MagicMock()
        
        # Create sample training data
        training_data = pd.DataFrame({
            'age': [30, 40, 50],
            'risk_tolerance': ['low', 'moderate', 'high'],
            'investment_horizon': ['short_term', 'medium_term', 'long_term'],
            'goal': ['retirement', 'education', 'house'],
            'income': [50000, 75000, 100000],
            'savings': [10000, 25000, 50000],
            'recommended_stocks': [30, 50, 70],
            'recommended_bonds': [60, 40, 20],
            'recommended_cash': [10, 10, 10]
        })
        
        # Train the models
        result = self.service.train(training_data)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that each recommender's train method was called
        self.service.advanced_recommender.train.assert_called_once_with(training_data)
        self.service.hybrid_recommender.train.assert_called_once_with(training_data)
        self.service.random_forest_recommender.train.assert_called_once_with(training_data)
        
        # Check the training result
        self.assertIn('advanced_recommender', result)
        self.assertIn('hybrid_recommender', result)
        self.assertIn('random_forest_recommender', result)
        self.assertEqual(result['advanced_recommender']['status'], 'success')
        self.assertEqual(result['hybrid_recommender']['status'], 'success')
        self.assertEqual(result['random_forest_recommender']['status'], 'success')
    
    @patch.object(InvestmentAIService, 'initialize')
    @patch.object(InvestmentAIService, '_ensure_initialized')
    @patch('investment.ml.utils.preprocessing.prepare_training_data')
    @patch.object(InvestmentAIService, 'train')
    def test_train_with_user_profiles(self, mock_train, mock_prepare_data, mock_ensure_initialized, mock_initialize):
        """Test training with user profiles."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock prepare_training_data
        mock_training_data = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        mock_prepare_data.return_value = mock_training_data
        
        # Mock train
        mock_train.return_value = {'status': 'success'}
        
        # Create sample user profiles and performances
        user_profiles = [MagicMock(), MagicMock()]
        portfolio_performances = [10.5, 8.2]
        
        # Train with user profiles
        result = self.service.train_with_user_profiles(user_profiles, portfolio_performances)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that prepare_training_data was called
        mock_prepare_data.assert_called_once_with(user_profiles, portfolio_performances)
        
        # Check that train was called with the prepared data
        mock_train.assert_called_once_with(mock_training_data)
        
        # Check the training result
        self.assertEqual(result['status'], 'success')
    
    @patch.object(InvestmentAIService, 'initialize')
    def test_get_recommender(self, mock_initialize):
        """Test getting a recommender."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the recommenders
        self.service.advanced_recommender = MagicMock()
        self.service.hybrid_recommender = MagicMock()
        self.service.random_forest_recommender = MagicMock()
        
        # Get each recommender
        advanced = self.service._get_recommender('advanced')
        hybrid = self.service._get_recommender('hybrid')
        random_forest = self.service._get_recommender('random_forest')
        
        # Check that we got the right recommenders
        self.assertIs(advanced, self.service.advanced_recommender)
        self.assertIs(hybrid, self.service.hybrid_recommender)
        self.assertIs(random_forest, self.service.random_forest_recommender)
        
        # Check that an unknown recommender type raises an error
        with self.assertRaises(ValueError):
            self.service._get_recommender('unknown')
    
    def test_assess_portfolio_risk(self):
        """Test assessing portfolio risk."""
        # Initialize the service
        self.service.initialize()
        
        # Assess portfolio risk
        risk_assessment = self.service._assess_portfolio_risk(self.mock_portfolio)
        
        # Check the risk assessment
        self.assertIn('risk_level', risk_assessment)
        self.assertIn('risk_score', risk_assessment)
        self.assertIn('description', risk_assessment)
    
    def test_calculate_diversification_score(self):
        """Test calculating diversification score."""
        # Initialize the service
        self.service.initialize()
        
        # Calculate diversification score
        score = self.service._calculate_diversification_score(self.mock_portfolio)
        
        # Check the score
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 10)
    
    def test_generate_improvement_suggestions(self):
        """Test generating improvement suggestions."""
        # Initialize the service
        self.service.initialize()
        
        # Create sample metrics
        metrics = {
            'risk_assessment': {
                'risk_level': 'high',
                'risk_score': 8
            },
            'diversification_score': 4,
            'returns_percentage': -2
        }
        
        # Generate improvement suggestions
        suggestions = self.service._generate_improvement_suggestions(self.mock_portfolio, metrics)
        
        # Check the suggestions
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)


if __name__ == '__main__':
    unittest.main()
