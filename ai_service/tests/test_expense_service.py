"""
Tests for the ExpenseAIService class.
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import pandas as pd
import joblib

from ai_service.expense_service import ExpenseAIService


class TestExpenseAIService(unittest.TestCase):
    """Test cases for the ExpenseAIService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = ExpenseAIService()
        
        # Sample data for testing
        self.sample_transaction = {
            'description': 'Grocery shopping at Walmart',
            'merchant': 'Walmart'
        }
        
        self.sample_transactions = [
            {'description': 'Grocery shopping at Walmart', 'category': 1},
            {'description': 'Amazon.com payment', 'category': 2},
            {'description': 'Starbucks coffee', 'category': 3},
            {'description': 'Gas station fill-up', 'category': 4},
            {'description': 'Monthly rent payment', 'category': 5}
        ]
    
    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('expenses.ml.model_training.ExpenseCategoryClassifier')
    def test_initialize(self, mock_classifier_class, mock_joblib_load, mock_path_exists):
        """Test initializing the service."""
        # Mock the classifier
        mock_classifier = MagicMock()
        mock_classifier.load_model.return_value = True
        mock_classifier_class.return_value = mock_classifier
        
        # Mock path exists
        mock_path_exists.return_value = True
        
        # Mock joblib load
        mock_joblib_load.return_value = MagicMock()
        
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(self.service.initialized)
        self.assertIsNotNone(self.service.classifier)
        
        # Check that the classifier was created and load_model was called
        mock_classifier_class.assert_called_once()
        mock_classifier.load_model.assert_called_once()
    
    @patch('os.path.exists')
    @patch('joblib.load')
    @patch('expenses.ml.model_training.ExpenseCategoryClassifier')
    def test_initialize_with_custom_model(self, mock_classifier_class, mock_joblib_load, mock_path_exists):
        """Test initializing the service with a custom model."""
        # Mock the classifier
        mock_classifier = MagicMock()
        mock_classifier.load_model.return_value = True
        mock_classifier_class.return_value = mock_classifier
        
        # Mock path exists
        mock_path_exists.return_value = True
        
        # Mock joblib load
        mock_custom_model = MagicMock()
        mock_vectorizer = MagicMock()
        mock_joblib_load.side_effect = [mock_custom_model, mock_vectorizer]
        
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(self.service.initialized)
        self.assertIsNotNone(self.service.classifier)
        self.assertIsNotNone(self.service.custom_model)
        self.assertIsNotNone(self.service.custom_vectorizer)
        
        # Check that joblib.load was called twice (for model and vectorizer)
        self.assertEqual(mock_joblib_load.call_count, 2)
    
    @patch.object(ExpenseAIService, 'initialize')
    @patch.object(ExpenseAIService, '_ensure_initialized')
    def test_predict(self, mock_ensure_initialized, mock_initialize):
        """Test predicting a category."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the classifier
        self.service.classifier = MagicMock()
        self.service.classifier.predict.return_value = {
            'category': 'Groceries',
            'subcategory': 'Supermarket',
            'confidence': 0.85
        }
        
        # Make a prediction
        result = self.service.predict(self.sample_transaction)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the classifier's predict method was called
        self.service.classifier.predict.assert_called_once_with(self.sample_transaction)
        
        # Check the prediction result
        self.assertEqual(result['category'], 'Groceries')
        self.assertEqual(result['subcategory'], 'Supermarket')
        self.assertEqual(result['confidence'], 0.85)
    
    @patch.object(ExpenseAIService, 'initialize')
    @patch.object(ExpenseAIService, '_ensure_initialized')
    def test_predict_with_custom_model(self, mock_ensure_initialized, mock_initialize):
        """Test predicting a category with the custom model."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the custom model and vectorizer
        self.service.custom_model = MagicMock()
        self.service.custom_vectorizer = MagicMock()
        
        # Mock transform and predict methods
        self.service.custom_vectorizer.transform.return_value = 'transformed_data'
        self.service.custom_model.predict.return_value = [1]  # Category ID
        self.service.custom_model.predict_proba.return_value = [[0.2, 0.8]]  # Probabilities
        
        # Mock Category model
        with patch('expenses.models.Category.objects.get') as mock_get:
            mock_category = MagicMock()
            mock_category.name = 'Groceries'
            mock_get.return_value = mock_category
            
            # Make a prediction
            result = self.service.predict_with_custom_model(self.sample_transaction)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the vectorizer's transform method was called
        self.service.custom_vectorizer.transform.assert_called_once()
        
        # Check that the model's predict and predict_proba methods were called
        self.service.custom_model.predict.assert_called_once_with('transformed_data')
        self.service.custom_model.predict_proba.assert_called_once_with('transformed_data')
        
        # Check the prediction result
        self.assertEqual(result['category'], 'Groceries')
        self.assertEqual(result['category_id'], 1)
        self.assertEqual(result['confidence'], 0.8)
    
    @patch.object(ExpenseAIService, 'initialize')
    @patch.object(ExpenseAIService, '_ensure_initialized')
    @patch('expenses.ml.utils.load_training_data')
    def test_train(self, mock_load_training_data, mock_ensure_initialized, mock_initialize):
        """Test training the model."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the classifier
        self.service.classifier = MagicMock()
        self.service.classifier.train_model.return_value = {
            'category_accuracy': 0.85,
            'subcategory_accuracy': 0.75
        }
        
        # Mock load_training_data
        mock_data = pd.DataFrame({'description': ['test'], 'category': [1]})
        mock_load_training_data.return_value = mock_data
        
        # Train the model
        result = self.service.train()
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that load_training_data was called
        mock_load_training_data.assert_called_once()
        
        # Check that the classifier's train_model method was called
        self.service.classifier.train_model.assert_called_once_with(mock_data)
        
        # Check the training result
        self.assertEqual(result['category_accuracy'], 0.85)
        self.assertEqual(result['subcategory_accuracy'], 0.75)
    
    @patch.object(ExpenseAIService, 'initialize')
    @patch.object(ExpenseAIService, '_ensure_initialized')
    @patch('joblib.dump')
    @patch('os.makedirs')
    def test_train_custom_model(self, mock_makedirs, mock_joblib_dump, mock_ensure_initialized, mock_initialize):
        """Test training a custom model."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock sklearn components
        with patch('sklearn.model_selection.train_test_split') as mock_train_test_split, \
             patch('sklearn.ensemble.RandomForestClassifier') as mock_rf_class, \
             patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_tfidf_class, \
             patch('sklearn.metrics.classification_report') as mock_classification_report, \
             patch('sklearn.metrics.confusion_matrix') as mock_confusion_matrix:
            
            # Mock train_test_split
            mock_train_test_split.return_value = (
                ['desc1', 'desc2'], ['desc3'], [1, 2], [1]  # X_train, X_test, y_train, y_test
            )
            
            # Mock TfidfVectorizer
            mock_vectorizer = MagicMock()
            mock_vectorizer.get_feature_names_out.return_value = ['word1', 'word2']
            mock_vectorizer.fit_transform.return_value = 'transformed_train'
            mock_vectorizer.transform.return_value = 'transformed_test'
            mock_tfidf_class.return_value = mock_vectorizer
            
            # Mock RandomForestClassifier
            mock_rf = MagicMock()
            mock_rf.feature_importances_ = [0.6, 0.4]
            mock_rf.score.return_value = 0.85
            mock_rf.predict.return_value = [1]
            mock_rf_class.return_value = mock_rf
            
            # Mock classification_report and confusion_matrix
            mock_classification_report.return_value = {'accuracy': 0.85}
            mock_confusion_matrix.return_value = [[1, 0], [0, 1]]
            
            # Train the custom model
            result = self.service.train_custom_model(self.sample_transactions)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that os.makedirs was called
        mock_makedirs.assert_called_once()
        
        # Check that joblib.dump was called twice (for model and vectorizer)
        self.assertEqual(mock_joblib_dump.call_count, 2)
        
        # Check that the model and vectorizer were saved
        self.assertIsNotNone(self.service.custom_model)
        self.assertIsNotNone(self.service.custom_vectorizer)
        
        # Check the training result
        self.assertEqual(result['accuracy'], 0.85)
        self.assertIn('report', result)
        self.assertIn('confusion_matrix', result)
        self.assertIn('feature_importance', result)
    
    @patch.object(ExpenseAIService, 'initialize')
    @patch.object(ExpenseAIService, '_ensure_initialized')
    def test_get_model_metrics(self, mock_ensure_initialized, mock_initialize):
        """Test getting model metrics."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the classifier
        self.service.classifier = MagicMock()
        self.service.classifier.get_metrics.return_value = {'accuracy': 0.85}
        
        # Mock the custom model and vectorizer
        self.service.custom_model = MagicMock()
        self.service.custom_vectorizer = MagicMock()
        self.service.custom_vectorizer.get_feature_names_out.return_value = ['word1', 'word2']
        
        # Get model metrics
        metrics = self.service.get_model_metrics()
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the classifier's get_metrics method was called
        self.service.classifier.get_metrics.assert_called_once()
        
        # Check the metrics
        self.assertIn('main_model', metrics)
        self.assertIn('custom_model', metrics)
        self.assertEqual(metrics['main_model']['accuracy'], 0.85)
        self.assertTrue(metrics['custom_model']['available'])
        self.assertEqual(metrics['custom_model']['feature_count'], 2)


if __name__ == '__main__':
    unittest.main()
