import re

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Transaction, Category, SubCategory
from .serializers import TransactionSerializer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os
import google.generativeai as genai


class TransactionListCreateView(generics.ListCreateAPIView):
  queryset = Transaction.objects.all()
  serializer_class = TransactionSerializer


  def create(self, request, *args, **kwargs):
      serializer = self.get_serializer(data=request.data)
      serializer.is_valid(raise_exception=True)
      # Category and subcategory are not set when creating a transaction
      serializer.save()
      headers = self.get_success_headers(serializer.data)
      return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)


class TransactionUpdateView(generics.RetrieveUpdateAPIView):
    queryset = Transaction.objects.all()
    serializer_class = TransactionSerializer



os.environ["GEMINI_API_KEY"] = "AIzaSyCq6_Uzz-GDrHzX5f_BCLD2PIZi3BoBsaY"
# Initialize Gemini API Key
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

_bert_classifier = None
def get_bert_classifier():
  global _bert_classifier
  if _bert_classifier is None:
      _bert_classifier =  model
  return _bert_classifier

#view to get category ID from name
class CategoryLookupView(APIView):
    def get(self, request):
        name = request.query_params.get('name')
        try:
            category = Category.objects.get(name=name)
            return Response({'id': category.id})
        except Category.DoesNotExist:
            return Response({'error': 'Category not found'}, status=404)

class SubCategoryLookupView(APIView):
    def get(self, request):
        name = request.query_params.get('name')
        category_id = request.query_params.get('category')
        try:
            subcategory = SubCategory.objects.get(
                name=name,
                category__id=category_id
            )
            return Response({'id': subcategory.id})
        except SubCategory.DoesNotExist:
            return Response({'error': 'Subcategory not found'}, status=404)


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
            model = genai.GenerativeModel('gemini-2.0-flash')

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

#Chatbot  View
class ChatbotView(APIView):
    def post(self, request, *args, **kwargs):
        prompt = request.data.get('prompt', '')
        # Use Gemini for initial predictions
        bert_classifier = get_bert_classifier()
        response = bert_classifier.generate_content(
            f"""You are a helpful financial assistant that provides useful information to users. 
             When asked a question you should always try to provide a full response that includes a title and also well formatted paragraphs with the information requested.
               Always answer as if you are speaking to a customer in a friendly way. Don't add space at the beginning of the sentence. At the end, ask whether clarification is needed or not.
              {prompt}"""
        )
        response_text = response.text
        # Remove leading and trailing whitespace
        response_text = response_text.strip()
        # Remove asterisks
        response_text = re.sub(r'\*+', '', response_text)
        return Response({ 'response': response_text })