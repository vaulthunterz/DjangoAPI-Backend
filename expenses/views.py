import re

from dotenv import load_dotenv
from rest_framework import generics, status, viewsets
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .firebase_auth import FirebaseAuthentication
from .models import Transaction, Category, SubCategory
from .serializers import TransactionSerializer, CategorySerializer, SubCategorySerializer
from .pagination import StandardResultsSetPagination, TransactionPagination, LargeResultsSetPagination
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os
import google.generativeai as genai
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
from .ml.model_training import ExpenseCategoryClassifier
from .gemini_prediction import GeminiPredictionView, get_gemini_model
from .custom_prediction import CustomModelPredictionView
from firebase_admin import auth



load_dotenv()
# Environment variable for the API key
# os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY"  #REPLACE KEY
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Load from .env
genai.configure(api_key='AIzaSyB465HZ8X-T5vqTfQuPBo4C_Qh66Q5PZgY')

# Initialize Gemini model (outside the class, for efficiency)
_gemini_model = None  # Use a leading underscore for "private" variables

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    return _gemini_model

class ModelMetrics:
    def __init__(self):
        self.last_trained = None
        self.training_count = 0
        self.accuracy = 0
        self.precision = {}
        self.recall = {}
        self.f1_scores = {}
        self.confusion_matrix = None
        self.num_transactions = 0
        self.num_categories = 0
        self.feature_importance = {}
        self.training_history = []

    def update(self, metrics_dict):
        self.last_trained = datetime.now()
        self.training_count += 1
        self.accuracy = metrics_dict.get('accuracy', 0)
        self.precision = metrics_dict.get('precision', {})
        self.recall = metrics_dict.get('recall', {})
        self.f1_scores = metrics_dict.get('f1_scores', {})
        self.confusion_matrix = metrics_dict.get('confusion_matrix', None)
        self.num_transactions = metrics_dict.get('num_transactions', 0)
        self.num_categories = metrics_dict.get('num_categories', 0)
        self.feature_importance = metrics_dict.get('feature_importance', {})

        # Add to training history
        history_entry = {
            'timestamp': self.last_trained.isoformat(),
            'accuracy': self.accuracy,
            'num_transactions': self.num_transactions,
            'num_categories': self.num_categories
        }
        self.training_history.append(history_entry)

# Global instance to store metrics
model_metrics = ModelMetrics()

class TransactionViewSet(viewsets.ModelViewSet):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer
    authentication_classes = [FirebaseAuthentication]
    permission_classes = [IsAuthenticated]
    pagination_class = TransactionPagination

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier = ExpenseCategoryClassifier()
        # Try to load existing model
        self.classifier.load_model()

    def get_queryset(self):
        queryset = Transaction.objects.filter(user=self.request.user)
        # Add filter for transaction type if requested
        transaction_type = self.request.query_params.get('type', None)
        if transaction_type is not None:
            is_expense = transaction_type.lower() == 'expense'
            queryset = queryset.filter(is_expense=is_expense)
        return queryset

    def perform_create(self, serializer):
        # Save the transaction with the current user
        transaction = serializer.save(user=self.request.user)

        # Automatically retrain the model
        self.retrain_model()

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        # Add Django user ID to context
        context['django_user_id'] = self.request.user.id
        return context

    def destroy(self, request, *args, **kwargs):
        try:
            # Log the request details
            print(f"Delete transaction request received. PK: {kwargs.get('pk')}")
            # Get the instance
            instance = self.get_object()
            print(f"Found transaction: ID={instance.id}, transaction_id={instance.transaction_id}")

            # Perform the deletion
            self.perform_destroy(instance)
            print(f"Successfully deleted transaction {kwargs.get('pk')}")

            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            print(f"Error deleting transaction: {str(e)}")
            return Response(
                {"detail": f"Error deleting transaction: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )

    def retrain_model(self):
        try:
            # Fetch transaction data
            transactions = Transaction.objects.all().select_related('category', 'subcategory')

            if len(transactions) < 10:
                return  # Not enough data to train

            # Prepare training data
            training_data = []

            for transaction in transactions:
                if transaction.category and transaction.subcategory:
                    training_data.append({
                        'description': transaction.description,
                        'merchant': transaction.merchant_name,
                        'category': transaction.category.name,
                        'subcategory': transaction.subcategory.name
                    })

            if len(training_data) < 2:
                return  # Not enough data

            # Convert to DataFrame
            df = pd.DataFrame(training_data)

            # Train model and get metrics
            metrics = self.classifier.train_model(df)

            # Update global metrics
            model_metrics.update({
                'accuracy': metrics['category_accuracy'],
                'num_transactions': len(training_data),
                'num_categories': len(metrics['unique_categories']),
                'feature_importance': metrics['category_feature_importance'],
                'precision': metrics.get('precision', {}),
                'recall': metrics.get('recall', {}),
                'f1_scores': metrics.get('f1_scores', {})
            })

        except Exception as e:
            print(f"Error retraining model: {str(e)}")

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):  # Read-only, as categories are usually predefined
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    pagination_class = StandardResultsSetPagination

class SubCategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = SubCategory.objects.all()
    serializer_class = SubCategorySerializer
    pagination_class = StandardResultsSetPagination
    def get_queryset(self):
        # Optionally filter by category if category_id is provided in the URL
        category_id = self.kwargs.get('category_pk')
        if category_id:
            return SubCategory.objects.filter(category_id=category_id)
        return SubCategory.objects.all()


#view to get category ID from name
class CategoryLookupView(APIView): #Keep this
    """
    Look up a Category by name and return its ID.
    """
    def get(self, request):
        name = request.query_params.get('name')
        if not name:
            return Response({'error': 'Category name is required'}, status=status.HTTP_400_BAD_REQUEST)
        try:
            category = Category.objects.get(name=name)
            return Response({'id': category.id})
        except Category.DoesNotExist:
            return Response({'error': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)

class SubCategoryLookupView(APIView): #Keep this
    """
    Look up a Subcategory by name and Category ID, return its ID.
    """
    def get(self, request):
        name = request.query_params.get('name')
        category_id = request.query_params.get('category')

        if not name or not category_id:
            return Response({'error': 'Subcategory name and category ID are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            subcategory = SubCategory.objects.get(name=name, category__id=category_id)
            return Response({'id': subcategory.id})
        except SubCategory.DoesNotExist:
            return Response({'error': 'Subcategory not found'}, status=status.HTTP_404_NOT_FOUND)


class CategoryPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '').strip()

        # Validate input
        if not description:
            return Response(
                { 'error': 'Description is required' },
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Fetch all categories and subcategories
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
                    { 'error': 'No categories or subcategories found' },
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Initialize Gemini model
            model = get_gemini_model() # Get the model

            # Create prompt for Gemini
            prompt = f"""
            Classify the following expense description into exactly one of these categories:
            {', '.join(labels)}

            Description: {description}

            Respond ONLY with the matching category name in the format "Category - Subcategory".
            If unsure, respond with "Unknown - Other".
            """

            # Generate prediction
            response = model.generate_content(prompt)
            predicted_category = response.text.strip()

            # Parse the predicted category and subcategory
            if ' - ' not in predicted_category:
                predicted_category = "Unknown - Other"  # Fallback for invalid format

            category_name, subcategory_name = predicted_category.split(' - ')

            # Lookup category and subcategory IDs
            try:
                category = Category.objects.get(name=category_name)
                subcategory = SubCategory.objects.get(
                    name=subcategory_name,
                    category=category
                )
            except (Category.DoesNotExist, SubCategory.DoesNotExist):
                # Fallback to "Unknown - Other" if predicted category doesn't exist
                category = Category.objects.get(name='Unknown')
                subcategory = SubCategory.objects.get(
                    name='Other',
                    category=category
                )

            # Return response with both names and IDs
            return Response({
                'category_name': f"{category.name} - {subcategory.name}",
                'category_id': category.id,
                'subcategory_id': subcategory.id
            })

        except Exception as e:
            return Response(
                { 'error': str(e) },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CustomModelPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '').strip()
        merchant = request.data.get('merchant', '').strip()

        # Validate input
        if not description or not merchant:
            return Response(
                {'error': 'Both description and merchant name are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'custom_model.joblib')
        vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'vectorizer.joblib')
        label_encoder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml', 'label_encoder.joblib')

        if not all(os.path.exists(path) for path in [model_path, vectorizer_path, label_encoder_path]):
            return Response(
                {'error': 'Custom model files not found. Please train the model first.'},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            # Load the model and preprocessing components
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            label_encoder = joblib.load(label_encoder_path)

            # Combine description and merchant for prediction
            combined_text = f"{description} {merchant}"
            features = vectorizer.transform([combined_text])

            # Get prediction and probability
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))

            print(f"Model predicted category: {prediction}")

            # Split the prediction into category and subcategory
            try:
                category_name, subcategory_name = prediction.split(' - ')
            except ValueError:
                print(f"Invalid prediction format: {prediction}")
                category_name, subcategory_name = "Unknown", "Other"

            print(f"Looking up category: {category_name}, subcategory: {subcategory_name}")

            try:
                # First try exact match
                category = Category.objects.filter(name__iexact=category_name).first()
                if not category:
                    # If no exact match, try case-insensitive contains
                    category = Category.objects.filter(name__icontains=category_name).first()

                if not category:
                    print(f"Category not found: {category_name}")
                    # Fallback to Unknown category
                    category = Category.objects.get(name='Unknown')
                    subcategory = SubCategory.objects.get(name='Other', category=category)
                else:
                    # Try to find matching subcategory
                    subcategory = SubCategory.objects.filter(
                        name__iexact=subcategory_name,
                        category=category
                    ).first()

                    if not subcategory:
                        # Try case-insensitive contains
                        subcategory = SubCategory.objects.filter(
                            name__icontains=subcategory_name,
                            category=category
                        ).first()

                    if not subcategory:
                        print(f"Subcategory not found: {subcategory_name}")
                        # Fallback to Other subcategory in the found category
                        subcategory = SubCategory.objects.get(name='Other', category=category)

            except (Category.DoesNotExist, SubCategory.DoesNotExist) as e:
                print(f"Error finding category/subcategory: {str(e)}")
                # Final fallback
                category = Category.objects.get(name='Unknown')
                subcategory = SubCategory.objects.get(name='Other', category=category)

            print(f"Final category: {category.name}, subcategory: {subcategory.name}")

            return Response({
                'category_name': category.name,
                'subcategory_name': subcategory.name,
                'category_id': category.id,
                'subcategory_id': subcategory.id,
                'confidence': confidence,
                'original_prediction': prediction  # Include the original prediction for debugging
            })

        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            return Response(
                {'error': f'Prediction error: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RetrainModelView(APIView):
    def load_baseline_data(self):
        """Load and process the baseline training data from CSV."""
        try:
            # Get the absolute path to the training data file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            csv_path = os.path.join(base_dir, 'backend', 'expenses', 'ml', 'training_data', 'transactions.csv')

            print(f"Attempting to load training data from: {csv_path}")

            if not os.path.exists(csv_path):
                # Try alternative path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(current_dir, 'ml', 'training_data', 'transactions.csv')
                print(f"First path not found, trying alternative path: {csv_path}")

                if not os.path.exists(csv_path):
                    raise Exception(f"Training data file not found at either expected location")

            print(f"Found training data file at: {csv_path}")
            df = pd.read_csv(csv_path)

            if df.empty:
                raise Exception("Training data file is empty")

            required_columns = ['description', 'category', 'subcategory', 'merchant']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise Exception(f"Training data file is missing required columns: {', '.join(missing_columns)}")

            # Prepare training data
            training_data = []
            labels = []

            for _, row in df.iterrows():
                # Combine description and merchant for features
                combined_text = f"{row['description']} {row['merchant']}"
                training_data.append(combined_text)
                # Combine category and subcategory for label
                label = f"{row['category']} - {row['subcategory']}"
                labels.append(label)

            if not training_data or not labels:
                raise Exception("No valid training data could be extracted from the file")

            print(f"Successfully loaded {len(training_data)} training examples")
            return training_data, labels
        except pd.errors.EmptyDataError:
            raise Exception("The CSV file is empty or corrupted")
        except pd.errors.ParserError as e:
            raise Exception(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading baseline data: {str(e)}")

    def load_db_transactions(self):
        """Load and process transactions from the database."""
        transactions = Transaction.objects.all().select_related('category', 'subcategory')

        training_data = []
        labels = []

        for transaction in transactions:
            if transaction.category and transaction.subcategory:
                combined_text = f"{transaction.description} {transaction.merchant_name}"
                training_data.append(combined_text)
                label = f"{transaction.category.name} - {transaction.subcategory.name}"
                labels.append(label)

        return training_data, labels, len(transactions)

    def train_model(self, training_data, labels):
        """Train the model with the given data."""
        if len(set(labels)) < 2:
            raise Exception('Not enough unique categories for training (minimum 2 required)')

        # Vectorize text data
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(training_data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

        # Train Random Forest classifier with balanced class weights
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)

        # Calculate metrics
        accuracy = clf.score(X_test, y_test)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()

        # Get feature importance
        feature_importance = dict(zip(vectorizer.get_feature_names_out(), clf.feature_importances_))

        # Save model components in the correct directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ml_dir = os.path.join(current_dir, 'ml')
        os.makedirs(ml_dir, exist_ok=True)

        joblib.dump(clf, os.path.join(ml_dir, 'custom_model.joblib'))
        joblib.dump(vectorizer, os.path.join(ml_dir, 'vectorizer.joblib'))

        # Extract category-specific metrics from the report
        metrics_dict = {
            'accuracy': accuracy,
            'precision': {},
            'recall': {},
            'f1_scores': {},
            'confusion_matrix': conf_matrix,
            'num_transactions': len(training_data),
            'num_categories': len(set(labels)),
            'feature_importance': feature_importance
        }

        # Extract precision, recall, and f1 from classification report
        for category in report.keys():
            if category not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_dict['precision'][category] = report[category]['precision']
                metrics_dict['recall'][category] = report[category]['recall']
                metrics_dict['f1_scores'][category] = report[category]['f1-score']

        # Update global metrics
        model_metrics.update(metrics_dict)

        return metrics_dict

    def post(self, request, *args, **kwargs):
        try:
            # Check if we should use baseline data
            use_baseline = request.data.get('use_baseline', False)

            if use_baseline:
                # Load and train with baseline data
                training_data, labels = self.load_baseline_data()
                training_results = self.train_model(training_data, labels)

                # Also save the label encoder for predictions
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ml_dir = os.path.join(current_dir, 'ml')
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                label_encoder.fit(labels)
                joblib.dump(label_encoder, os.path.join(ml_dir, 'label_encoder.joblib'))

                return Response({
                    'message': 'Baseline model trained successfully',
                    'accuracy': training_results['accuracy'],
                    'num_transactions': training_results['num_transactions'],
                    'num_categories': training_results['num_categories'],
                    'precision': training_results['precision'],
                    'recall': training_results['recall'],
                    'f1_scores': training_results['f1_scores'],
                    'confusion_matrix': training_results['confusion_matrix'],
                    'feature_importance': training_results['feature_importance']
                })
            else:
                # Load transactions from database
                db_data, db_labels, num_transactions = self.load_db_transactions()

                if num_transactions < 10:
                    return Response(
                        {'error': 'Not enough data to train model (minimum 10 transactions required). Use baseline training instead.'},
                        status=status.HTTP_400_BAD_REQUEST
                    )

                # Train with database transactions
                training_results = self.train_model(db_data, db_labels)

                # Update label encoder for predictions
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ml_dir = os.path.join(current_dir, 'ml')
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                label_encoder.fit(db_labels)
                joblib.dump(label_encoder, os.path.join(ml_dir, 'label_encoder.joblib'))

                return Response({
                    'message': 'Custom model retrained successfully with database transactions',
                    'accuracy': training_results['accuracy'],
                    'num_transactions': training_results['num_transactions'],
                    'num_categories': training_results['num_categories'],
                    'precision': training_results['precision'],
                    'recall': training_results['recall'],
                    'f1_scores': training_results['f1_scores'],
                    'confusion_matrix': training_results['confusion_matrix'],
                    'feature_importance': training_results['feature_importance']
                })

        except Exception as e:
            return Response(
                {'error': f'Training error: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

#Chatbot  View
class ChatbotView(APIView):
    def post(self, request, *args, **kwargs):
        prompt = request.data.get('prompt', '').strip()

        # Handle missing prompt
        if not prompt:
            return Response({ 'error': 'Prompt is required' }, status=400)

        try:
            # Get Gemini model
            gemini_model = get_gemini_model()

            # Generate response from the model
            response = gemini_model.generate_content(
                f"""You are a helpful financial assistant that provides useful information to users.
                When asked a question you should always try to provide a full response that includes a title and also well-formatted paragraphs with the information requested.
                Always answer as if you are speaking to a customer in a friendly way. Don't add space at the beginning of the sentence.
                At the end, ask whether clarification is needed or not.
                {prompt}"""
            )

            # Extract text response
            response_text = response.text.strip()

            # Remove asterisks (if they exist)
            response_text = re.sub(r'\*+', '', response_text)

            return Response({ 'response': response_text })

        except Exception as e:
            return Response({ 'error': str(e) }, status=500)

class ModelMetricsView(APIView):
    """
    View to get model performance metrics and training history.
    """
    def get(self, request, *args, **kwargs):
        return Response({
            'last_trained': model_metrics.last_trained,
            'training_count': model_metrics.training_count,
            'accuracy': model_metrics.accuracy,
            'precision': model_metrics.precision,
            'recall': model_metrics.recall,
            'f1_scores': model_metrics.f1_scores,
            'confusion_matrix': model_metrics.confusion_matrix,
            'num_transactions': model_metrics.num_transactions,
            'num_categories': model_metrics.num_categories,
            'feature_importance': dict(sorted(
                model_metrics.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20]),  # Top 20 important features
            'training_history': model_metrics.training_history
        })

class ChangePasswordView(APIView):
    authentication_classes = [FirebaseAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            current_password = request.data.get('current_password')
            new_password = request.data.get('new_password')
            confirm_password = request.data.get('confirm_password')

            if not all([current_password, new_password, confirm_password]):
                return Response(
                    {'error': 'All fields are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            if new_password != confirm_password:
                return Response(
                    {'error': 'New passwords do not match'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                # Get the Firebase user
                user = auth.get_user_by_email(request.user.email)

                # Update the password directly
                # Since the user is already authenticated through FirebaseAuthentication,
                # we can trust that they are the legitimate user
                auth.update_user(user.uid, password=new_password)

                return Response(
                    {'message': 'Password updated successfully'},
                    status=status.HTTP_200_OK
                )
            except Exception as e:
                print(f"Firebase operation failed: {str(e)}")
                return Response(
                    {'error': f'Error updating password: {str(e)}'},
                    status=status.HTTP_400_BAD_REQUEST
                )

        except Exception as e:
            print(f"General error in password change: {str(e)}")
            return Response(
                {'error': f'Error processing request: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )