import os
import joblib
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import Category, SubCategory

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