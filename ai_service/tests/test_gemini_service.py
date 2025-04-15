"""
Tests for the GeminiAIService class.
"""
import unittest
from unittest.mock import patch, MagicMock

from ai_service.gemini_service import GeminiAIService


class TestGeminiAIService(unittest.TestCase):
    """Test cases for the GeminiAIService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = GeminiAIService()
        
        # Sample data for testing
        self.sample_prompt = "What is the best way to save money?"
        self.sample_description = "Grocery shopping at Walmart"
        self.sample_categories = ["Groceries", "Shopping", "Entertainment", "Bills"]
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('django.conf.settings')
    def test_initialize(self, mock_settings, mock_generative_model, mock_configure):
        """Test initializing the service."""
        # Mock settings
        mock_settings.GOOGLE_AI_STUDIO_KEY = 'test_api_key'
        
        # Mock GenerativeModel
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertTrue(result)
        self.assertTrue(self.service.initialized)
        self.assertIsNotNone(self.service.model)
        self.assertEqual(self.service.api_key, 'test_api_key')
        
        # Check that configure was called with the API key
        mock_configure.assert_called_once_with(api_key='test_api_key')
        
        # Check that GenerativeModel was called with the model name
        mock_generative_model.assert_called_once_with('gemini-2.0-flash')
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @patch('django.conf.settings')
    def test_initialize_no_api_key(self, mock_settings, mock_generative_model, mock_configure):
        """Test initializing the service with no API key."""
        # Mock settings with no API key
        mock_settings.GOOGLE_AI_STUDIO_KEY = None
        
        # Initialize the service
        result = self.service.initialize()
        
        # Check initialization result
        self.assertFalse(result)
        self.assertFalse(self.service.initialized)
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_predict(self, mock_ensure_initialized, mock_initialize):
        """Test predicting with Gemini."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test response"
        self.service.model.generate_content.return_value = mock_response
        
        # Make a prediction
        result = self.service.predict({'prompt': self.sample_prompt})
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once_with(self.sample_prompt)
        
        # Check the prediction result
        self.assertEqual(result['text'], "This is a test response")
        self.assertEqual(result['model'], 'gemini-2.0-flash')
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_predict_with_description(self, mock_ensure_initialized, mock_initialize):
        """Test predicting with a description instead of a prompt."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a test response"
        self.service.model.generate_content.return_value = mock_response
        
        # Make a prediction with a description
        result = self.service.predict({'description': self.sample_description})
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once_with(self.sample_description)
        
        # Check the prediction result
        self.assertEqual(result['text'], "This is a test response")
        self.assertEqual(result['model'], 'gemini-2.0-flash')
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_predict_category(self, mock_ensure_initialized, mock_initialize):
        """Test predicting a category."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Groceries - Supermarket"
        self.service.model.generate_content.return_value = mock_response
        
        # Predict a category
        result = self.service.predict_category(self.sample_description, self.sample_categories)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once()
        
        # Check the prediction result
        self.assertEqual(result['category'], "Groceries")
        self.assertEqual(result['subcategory'], "Supermarket")
        self.assertIn('confidence', result)
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_predict_category_invalid_format(self, mock_ensure_initialized, mock_initialize):
        """Test predicting a category with an invalid format."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Groceries"  # Missing the separator
        self.service.model.generate_content.return_value = mock_response
        
        # Predict a category
        result = self.service.predict_category(self.sample_description, self.sample_categories)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once()
        
        # Check the prediction result
        self.assertEqual(result['category'], "Unknown")
        self.assertEqual(result['subcategory'], "Other")
        self.assertIn('confidence', result)
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_get_chatbot_response(self, mock_ensure_initialized, mock_initialize):
        """Test getting a chatbot response."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Here's how you can save money..."
        self.service.model.generate_content.return_value = mock_response
        
        # Get a chatbot response
        result = self.service.get_chatbot_response(self.sample_prompt)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once()
        
        # Check the response
        self.assertEqual(result['text'], "Here's how you can save money...")
        self.assertEqual(result['model'], 'gemini-2.0-flash')
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_get_chatbot_response_with_context(self, mock_ensure_initialized, mock_initialize):
        """Test getting a chatbot response with conversation context."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock the model
        self.service.model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Here's how you can save money..."
        self.service.model.generate_content.return_value = mock_response
        
        # Create conversation context
        context = [
            {'role': 'user', 'content': 'How can I manage my finances?'},
            {'role': 'assistant', 'content': 'There are several ways to manage your finances...'}
        ]
        
        # Get a chatbot response with context
        result = self.service.get_chatbot_response(self.sample_prompt, context)
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that the model's generate_content method was called
        self.service.model.generate_content.assert_called_once()
        
        # Check the response
        self.assertEqual(result['text'], "Here's how you can save money...")
        self.assertEqual(result['model'], 'gemini-2.0-flash')
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_train(self, mock_ensure_initialized, mock_initialize):
        """Test training (which is not applicable for Gemini)."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Try to train the model
        result = self.service.train({'data': 'test'})
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check the result
        self.assertEqual(result['status'], 'not_applicable')
        self.assertIn('message', result)
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    @patch('google.generativeai.GenerativeModel')
    def test_set_model(self, mock_generative_model, mock_ensure_initialized, mock_initialize):
        """Test setting a different model."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Mock GenerativeModel
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        
        # Set a different model
        result = self.service.set_model('gemini-2.0-pro')
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check that GenerativeModel was called with the new model name
        mock_generative_model.assert_called_once_with('gemini-2.0-pro')
        
        # Check the result
        self.assertTrue(result)
        self.assertEqual(self.service.model_name, 'gemini-2.0-pro')
        self.assertIn('gemini-2.0-pro', self.service.models)
    
    @patch.object(GeminiAIService, 'initialize')
    @patch.object(GeminiAIService, '_ensure_initialized')
    def test_set_invalid_model(self, mock_ensure_initialized, mock_initialize):
        """Test setting an invalid model."""
        # Mock initialize to return True
        mock_initialize.return_value = True
        
        # Initialize the service
        self.service.initialize()
        
        # Try to set an invalid model
        result = self.service.set_model('invalid-model')
        
        # Check that _ensure_initialized was called
        mock_ensure_initialized.assert_called_once()
        
        # Check the result
        self.assertFalse(result)
        self.assertEqual(self.service.model_name, 'gemini-2.0-flash')  # Should not change


if __name__ == '__main__':
    unittest.main()
