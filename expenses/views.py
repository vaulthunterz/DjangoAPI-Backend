"""
API views for the expenses app.
"""
from rest_framework import status, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

# Import Firebase auth for password changes
from firebase_admin import auth

# Import models and serializers
from .models import Transaction, Category, SubCategory
from .serializers import TransactionSerializer, CategorySerializer, SubCategorySerializer
from .pagination import StandardResultsSetPagination, TransactionPagination

# Model metrics are now handled by the AI service

class TransactionViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing transactions.

    This viewset provides CRUD operations for transactions, with automatic
    categorization using machine learning models.

    list:
        Returns a paginated list of transactions for the authenticated user.
        Can be filtered by transaction type using the 'type' query parameter.

    create:
        Creates a new transaction and automatically retrains the ML model.

    retrieve:
        Returns details of a specific transaction.

    update:
        Updates an existing transaction.

    partial_update:
        Partially updates an existing transaction.

    delete:
        Deletes a transaction.
    """
    queryset = Transaction.objects.all().order_by('-time_of_transaction')
    serializer_class = TransactionSerializer
    pagination_class = TransactionPagination
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        Filter transactions by the authenticated user.
        """
        queryset = Transaction.objects.filter(user=self.request.user).order_by('-time_of_transaction')

        # Filter by transaction type (expense or income) if provided
        transaction_type = self.request.query_params.get('type', None)
        if transaction_type:
            is_expense = transaction_type.lower() == 'expense'
            queryset = queryset.filter(is_expense=is_expense)

        return queryset

    def perform_create(self, serializer):
        """
        Save the transaction with the current user.
        """
        serializer.save(user=self.request.user)

        # Note: Model retraining is now handled by the AI service

    def get_serializer_context(self):
        """
        Add the user ID to the serializer context.
        """
        context = super().get_serializer_context()
        context['django_user_id'] = self.request.user.id
        return context

    def destroy(self, *args, **kwargs):
        try:
            # Log the request details
            print(f"Delete transaction request received. PK: {kwargs.get('pk')}")

            # Get the transaction
            transaction = self.get_object()

            # Log the transaction details
            print(f"Found transaction: {transaction.id} - {transaction.description}")

            # Delete the transaction
            transaction.delete()

            # Return success response
            return Response(status=status.HTTP_204_NO_CONTENT)

        except Exception as e:
            # Log the error
            print(f"Error deleting transaction: {str(e)}")

            # Return error response
            return Response(
                {'error': f'Failed to delete transaction: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CategoryViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing expense categories.
    """
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsAuthenticated]


class SubCategoryViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing expense subcategories.
    """
    queryset = SubCategory.objects.all()
    serializer_class = SubCategorySerializer
    pagination_class = StandardResultsSetPagination
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        Filter subcategories by category if provided.
        """
        queryset = SubCategory.objects.all()

        # Filter by category if provided
        category_id = self.request.query_params.get('category', None)
        if category_id:
            queryset = queryset.filter(category_id=category_id)

        return queryset


class CategoryLookupView(APIView):
    """
    API endpoint for looking up categories.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """
        Return a list of all categories.
        """
        categories = Category.objects.all()
        data = [{'id': category.id, 'name': category.name} for category in categories]
        return Response(data)


class SubCategoryLookupView(APIView):
    """
    API endpoint for looking up subcategories.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """
        Return a list of subcategories, filtered by category if provided.
        """
        category_id = request.query_params.get('category', None)

        if category_id:
            try:
                category = Category.objects.get(id=category_id)
                subcategories = SubCategory.objects.filter(category=category)
                data = [{'id': subcategory.id, 'name': subcategory.name, 'category_id': category.id} for subcategory in subcategories]
                return Response(data)
            except Category.DoesNotExist:
                return Response({'error': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)
        else:
            subcategories = SubCategory.objects.all()
            data = [{'id': subcategory.id, 'name': subcategory.name, 'category_id': subcategory.category.id} for subcategory in subcategories]
            return Response(data)

    def post(self, request, *args, **kwargs):
        """
        Look up a subcategory by name and category.
        """
        name = request.data.get('name', '').strip()
        category_id = request.data.get('category_id')

        if not name or not category_id:
            return Response({'error': 'Name and category_id are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            category = Category.objects.get(id=category_id)
            try:
                subcategory = SubCategory.objects.get(name=name, category=category)
                return Response({'id': subcategory.id, 'name': subcategory.name, 'category_id': category.id})
            except SubCategory.DoesNotExist:
                return Response({'error': 'Subcategory not found'}, status=status.HTTP_404_NOT_FOUND)
        except Category.DoesNotExist:
            return Response({'error': 'Category not found'}, status=status.HTTP_404_NOT_FOUND)


class ChangePasswordView(APIView):
    """
    API endpoint for changing a user's password.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """
        Change the user's password.
        """
        old_password = request.data.get('old_password', '')
        new_password = request.data.get('new_password', '')

        if not old_password or not new_password:
            return Response({'error': 'Old password and new password are required'}, status=status.HTTP_400_BAD_REQUEST)

        # Get the Firebase user ID from the request
        firebase_user_id = request.user.username

        try:
            # Update the password in Firebase
            auth.update_user(firebase_user_id, password=new_password)

            return Response({'message': 'Password updated successfully'})

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
