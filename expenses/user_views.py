"""
User-related views for the expenses app.
"""
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth.models import User
from firebase_admin import auth

from .serializers import UserSerializer

class UserProfileView(APIView):
    """
    API endpoint for managing user profile information.

    This view provides operations to retrieve and update user profile information
    for the authenticated user.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        """
        Get the current user's profile information.
        """
        user = request.user
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def put(self, request, *args, **kwargs):
        """
        Update the current user's profile information.
        """
        print("User profile update request received")
        print(f"Request data: {request.data}")
        print(f"Authenticated user: {request.user.username}")

        user = request.user
        data = request.data

        # Extract fields that can be updated
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        email = data.get('email')

        print(f"Updating user profile - First name: {first_name}, Last name: {last_name}, Email: {email}")

        # Update the user object
        if first_name is not None:
            user.first_name = first_name
        if last_name is not None:
            user.last_name = last_name
        if email is not None and email != user.email:
            # Check if email is already in use
            if User.objects.filter(email=email).exclude(id=user.id).exists():
                print(f"Email {email} is already in use by another account")
                return Response(
                    {'error': 'Email is already in use by another account'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            user.email = email

        # Save the user object
        try:
            user.save()
            print(f"User {user.username} updated successfully in Django")
        except Exception as e:
            print(f"Error saving user: {str(e)}")
            return Response(
                {'error': f'Error saving user: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # If Firebase UID is provided, try to update Firebase user as well
        firebase_uid = data.get('firebase_uid')
        if firebase_uid:
            try:
                print(f"Updating Firebase user with UID: {firebase_uid}")
                # Update Firebase user display name
                display_name = f"{first_name} {last_name}".strip()
                if display_name:
                    auth.update_user(
                        firebase_uid,
                        display_name=display_name,
                        email=email if email else None
                    )
                    print(f"Firebase user updated with display name: {display_name}")
            except Exception as e:
                # Log the error but don't fail the request
                print(f"Error updating Firebase user: {str(e)}")
        else:
            print("No Firebase UID provided, skipping Firebase update")

        # Return the updated user object
        serializer = UserSerializer(user)
        print(f"Returning updated user data: {serializer.data}")
        return Response(serializer.data)
