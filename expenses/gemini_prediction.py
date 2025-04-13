import google.generativeai as genai
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Category, SubCategory

# Initialize Gemini model (outside the class, for efficiency)
_gemini_model = None  # Use a leading underscore for "private" variables

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    return _gemini_model

class GeminiPredictionView(APIView):
    """
    API endpoint for predicting expense categories using Google's Gemini AI model.

    This view takes a transaction description and uses Gemini to predict the most
    appropriate category and subcategory based on existing categories in the system.
    """
    def post(self, request, *args, **kwargs):
        """
        Predict category for a transaction description using Gemini AI.

        Parameters:
        - description: The transaction description text to categorize

        Returns:
        - category: The predicted category name
        - subcategory: The predicted subcategory name
        - confidence: Confidence score of the prediction (0-1)
        """

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