"""
API views for the AI service.
"""
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .analysis import ModelAnalyzer
from expenses.ml.config import FEATURE_WEIGHTS, MAX_FEATURES, NGRAM_RANGE, TEST_SIZE, USE_ENSEMBLE, update_feature_weights
from django.shortcuts import get_object_or_404
import logging
import joblib

# Configure logging
logger = logging.getLogger(__name__)

from .factory import (
    get_expense_ai_service,
    get_investment_ai_service,
    get_gemini_ai_service,
    ai_service_factory
)

from expenses.models import Category, SubCategory
from investment.models import Portfolio, UserProfile


class ExpensePredictionView(APIView):
    """
    API endpoint for predicting expense categories.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '').strip()
        merchant = request.data.get('merchant', '').strip()

        # Validate input
        if not description:
            return Response(
                {'error': 'Description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the expense AI service
            expense_service = get_expense_ai_service()

            # Prepare input data
            input_data = {
                'description': description,
                'merchant': merchant
            }

            # Make prediction
            prediction = expense_service.predict(input_data)

            return Response(prediction)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExpenseCustomPredictionView(APIView):
    """
    API endpoint for predicting expense categories using the custom model.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '').strip()
        merchant = request.data.get('merchant', '').strip()

        # Validate input
        if not description:
            return Response(
                {'error': 'Description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the expense AI service
            expense_service = get_expense_ai_service()

            # Prepare input data
            input_data = {
                'description': description,
                'merchant': merchant
            }

            # Make prediction using the custom model
            prediction_result = expense_service.predict_with_custom_model(input_data)

            # Format the response to match the expected format
            prediction = {
                'category_name': prediction_result.get('category', ''),
                'subcategory_name': prediction_result.get('subcategory', ''),
                'confidence': prediction_result.get('confidence', 0.85),
                'category_id': prediction_result.get('category_id'),
                'subcategory_id': prediction_result.get('subcategory_id'),
                'model_type': 'custom'
            }

            return Response(prediction)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExpenseModelMetricsView(APIView):
    """
    API endpoint for getting expense model metrics.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            # Get the expense AI service
            expense_service = get_expense_ai_service()

            # Get model metrics
            metrics = expense_service.get_model_metrics()

            return Response(metrics)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExpenseModelTrainingView(APIView):
    """
    API endpoint for training the expense model.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        use_baseline = request.data.get('use_baseline', False)
        reset_model = request.data.get('reset_model', False)

        try:
            # Get the expense AI service
            expense_service = get_expense_ai_service()

            # Train the model with the provided options
            if reset_model:
                result = expense_service.reset_model()
            elif use_baseline:
                result = expense_service.train_with_baseline()
            else:
                result = expense_service.train()

            return Response(result)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GeminiPredictionView(APIView):
    """
    API endpoint for generating predictions using Gemini AI.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '').strip()

        # Validate input
        if not description:
            return Response(
                {'error': 'Description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the Gemini AI service
            gemini_service = get_gemini_ai_service()

            # Get categories from the database
            from expenses.models import Category, SubCategory
            categories = Category.objects.all()
            subcategories = SubCategory.objects.all()

            # Create labels in "Category - Subcategory" format
            labels = [
                f"{category.name} - {subcategory.name}"
                for category in categories
                for subcategory in subcategories.filter(category=category)
            ]

            if not labels:
                return Response(
                    {'error': 'No categories or subcategories found'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Make prediction using the predict_category method
            prediction_result = gemini_service.predict_category(description, labels)

            # Create a new prediction object with the expected keys
            prediction = {
                'category_name': prediction_result['category'],
                'subcategory_name': prediction_result['subcategory'],
                'confidence': prediction_result.get('confidence', 0.85),  # Default confidence if not provided
                'model_type': 'gemini'
            }

            # Get category and subcategory IDs
            try:
                category_name = prediction_result['category']
                subcategory_name = prediction_result['subcategory']

                category = Category.objects.get(name=category_name)
                subcategory = SubCategory.objects.get(name=subcategory_name, category=category)

                # Add IDs to the response
                prediction['category_id'] = category.id
                prediction['subcategory_id'] = subcategory.id
            except (Category.DoesNotExist, SubCategory.DoesNotExist):
                # If the predicted category doesn't exist, use Unknown - Other
                try:
                    category = Category.objects.get(name='Unknown')
                    subcategory = SubCategory.objects.get(name='Other', category=category)

                    prediction['category_id'] = category.id
                    prediction['subcategory_id'] = subcategory.id
                except (Category.DoesNotExist, SubCategory.DoesNotExist):
                    # If Unknown - Other doesn't exist, don't add IDs
                    pass

            return Response(prediction)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GeminiChatView(APIView):
    """
    API endpoint for generating chatbot responses using Gemini AI.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        message = request.data.get('message', '').strip()
        context = request.data.get('context', [])

        # Validate input
        if not message:
            return Response(
                {'error': 'Message is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the Gemini AI service
            gemini_service = get_gemini_ai_service()

            # Get chatbot response
            response = gemini_service.get_chatbot_response(message, context)

            return Response(response)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class InvestmentPredictionView(APIView):
    """
    API endpoint for generating investment recommendations.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        user_profile = request.data.get('user_profile', {})
        recommender_type = request.data.get('recommender_type', 'advanced')

        # Validate input
        if not user_profile:
            return Response(
                {'error': 'User profile is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the investment AI service
            investment_service = get_investment_ai_service()

            # Prepare input data
            input_data = {
                'user_profile': user_profile,
                'recommender_type': recommender_type
            }

            # Make prediction
            recommendations = investment_service.predict(input_data)

            return Response(recommendations)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class InvestmentPortfolioAnalysisView(APIView):
    """
    API endpoint for analyzing investment portfolios.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, portfolio_id, *args, **kwargs):
        try:
            # Get the portfolio
            portfolio = get_object_or_404(Portfolio, id=portfolio_id)

            # Get the investment AI service
            investment_service = get_investment_ai_service()

            # Analyze the portfolio
            analysis = investment_service.analyze_portfolio(portfolio)

            return Response(analysis)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class InvestmentExpenseRecommendationsView(APIView):
    """
    API endpoint for generating expense-based investment recommendations.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, user_id, *args, **kwargs):
        try:
            # Get the user profile
            user_profile = get_object_or_404(UserProfile, id=user_id)

            # Get the investment AI service
            investment_service = get_investment_ai_service()

            # Get expense-based recommendations
            recommendations = investment_service.get_expense_based_recommendations(user_profile)

            return Response(recommendations)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AIServiceInfoView(APIView):
    """
    API endpoint for getting information about the AI services.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        try:
            # Get information about all AI services
            info = ai_service_factory.get_service_info()

            return Response(info)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExpenseModelAnalysisView(APIView):
    """
    API endpoint for analyzing expense model performance and misclassifications.

    This endpoint allows you to submit a batch of transactions with known categories
    and analyze how well the model performs on them, with detailed information about
    which categories are most frequently misclassified.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # Get transactions from request
        transactions = request.data.get('transactions', [])
        model_type = request.data.get('model_type', 'custom')

        # Validate input
        if not transactions:
            return Response(
                {'error': 'No transactions provided for analysis'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the expense AI service
            expense_service = get_expense_ai_service()

            # Create analyzer
            analyzer = ModelAnalyzer(expense_service)

            # Analyze misclassifications
            analysis_results = analyzer.analyze_misclassifications(
                transactions=transactions,
                model_type=model_type
            )

            return Response(analysis_results)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ExpenseModelConfigView(APIView):
    """
    API endpoint for getting and updating expense model configuration.

    This endpoint allows you to view and modify the model configuration,
    such as feature weights, and optionally retrain the model with the new settings.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """Get the current model configuration"""
        try:
            # Get metadata from a trained model if available
            expense_service = get_expense_ai_service()
            model_version = "Unknown"

            if expense_service and expense_service.classifier:
                # Try to get feature weights from the classifier
                feature_weights = getattr(expense_service.classifier, 'feature_weights', FEATURE_WEIGHTS)
                use_ensemble = getattr(expense_service.classifier, 'use_ensemble', USE_ENSEMBLE)

                # Try to get model version from metadata
                if hasattr(expense_service.classifier, 'models_dir'):
                    import os
                    import joblib
                    metadata_path = os.path.join(expense_service.classifier.models_dir, 'metadata.joblib')
                    if os.path.exists(metadata_path):
                        try:
                            metadata = joblib.load(metadata_path)
                            if 'model_version' in metadata:
                                model_version = metadata['model_version']
                        except Exception as e:
                            logger.error(f"Error loading metadata: {str(e)}")
            else:
                # Use default values from config
                feature_weights = FEATURE_WEIGHTS
                use_ensemble = USE_ENSEMBLE

            # Return the configuration
            config = {
                'feature_weights': feature_weights,
                'max_features': MAX_FEATURES,
                'ngram_range': NGRAM_RANGE,
                'test_size': TEST_SIZE,
                'use_ensemble': use_ensemble,
                'model_version': model_version
            }

            return Response(config)

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def post(self, request, *args, **kwargs):
        """Update the model configuration"""
        try:
            # Get feature weights from request
            feature_weights = request.data.get('feature_weights')
            retrain_model = request.data.get('retrain_model', False)

            # Validate input
            if not feature_weights:
                return Response(
                    {'error': 'No feature weights provided'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Update feature weights in config
            update_feature_weights(
                description_weight=feature_weights.get('description'),
                merchant_weight=feature_weights.get('merchant')
            )

            # Retrain model if requested
            if retrain_model:
                # Get the expense AI service
                expense_service = get_expense_ai_service()

                # Start training in a background thread
                import threading

                def train_model_task():
                    try:
                        # Train the model
                        expense_service.train()
                        logger.info("Model training completed successfully")
                    except Exception as e:
                        logger.error(f"Error training model: {str(e)}")

                # Start training in a background thread
                training_thread = threading.Thread(target=train_model_task)
                training_thread.daemon = True
                training_thread.start()

                return Response({
                    'message': 'Feature weights updated and model training started',
                    'feature_weights': FEATURE_WEIGHTS
                })
            else:
                return Response({
                    'message': 'Feature weights updated',
                    'feature_weights': FEATURE_WEIGHTS
                })

        except Exception as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
