from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Transaction, SubCategory, Category
from .serializers import TransactionSerializer
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os


class TransactionListCreateView(generics.ListCreateAPIView):
 queryset = Transaction.objects.all()
 serializer_class = TransactionSerializer

# Zero-shot classification model for initial predictions
_bert_classifier = None
def get_bert_classifier():
     global _bert_classifier
     if _bert_classifier is None:
         _bert_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
     return _bert_classifier


class CategoryPredictionView(APIView):
    def post(self, request, *args, **kwargs):
        description = request.data.get('description', '')
        # Fetch categories and subcategories
        categories = Category.objects.all()
        subcategories = SubCategory.objects.all()

        # Create labels
        labels = [
            f"{category.name} - {subcategory.name}"
            for category in categories
            for subcategory in subcategories if subcategory.category == category
        ]

        # Use BERT for initial predictions
        # Zero-shot classification model for initial predictions
        bert_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        predicted = bert_classifier(description, candidate_labels=labels)
        category = predicted['labels'][0]
        return Response({'category': category})

class CustomModelPredictionView(APIView):
 def post(self, request, *args, **kwargs):
    description = request.data.get('description', '')
    model_path = 'expenses/custom_model.joblib'
    vectorizer_path = 'expenses/vectorizer.joblib'

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
       return Response({'category': 'No custom model trained yet'}, status=status.HTTP_404_NOT_FOUND)
    try:
       model = joblib.load(model_path)
       vectorizer = joblib.load(vectorizer_path)
       description_tfidf = vectorizer.transform([description])
       category = model.predict(description_tfidf)[0]
       return Response({'category': category})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class RetrainModelView(APIView):
    def post(self, request, *args, **kwargs):
        # Fetches transaction data
        transactions = Transaction.objects.all()
        if len(transactions) < 10:
            return Response({'message': 'Not enough data to train model'}, status=status.HTTP_400_BAD_REQUEST)
        data = transactions.values('description', 'category')
        df = pd.DataFrame(list(data))

        # Prepare features and labels
        X = df['description'].fillna('')
        y = df['category'].fillna('unknown')
        if len(set(y)) < 2:
            return Response({'message': 'Not enough categories for training'}, status=status.HTTP_400_BAD_REQUEST)

        # Vectorize text data
        vectorizer = TfidfVectorizer(max_features=1000)
        X_vec = vectorizer.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

        # Train Random Forest classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Save model and vectorizer
        model_path = 'expenses/custom_model.joblib'
        vectorizer_path = 'expenses/vectorizer.joblib'
        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)

        return Response({'message': 'Custom model retrained successfully'})

class TestAPIView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({ 'message': 'Test API is working' })