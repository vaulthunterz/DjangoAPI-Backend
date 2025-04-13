import firebase_admin
from firebase_admin import credentials, auth
from django.conf import settings
from django.contrib.auth.models import User
from rest_framework import authentication
from rest_framework import exceptions
import os


print("----- firebase_auth.py is being loaded -----")
# --- Firebase Initialization (IMPORTANT: Only initialize once!) ---

# Define the project ID from your Firebase project
FIREBASE_PROJECT_ID = "expensecategorization-auth"  # This matches your frontend config

try:
    # First try to get an existing Firebase app instance
    firebase_admin.get_app()
    print("Firebase already initialized.")
except ValueError:
    # If no app exists, initialize with service account
    cred_path = os.path.join(settings.BASE_DIR, 'serviceAccountKey.json')
    if os.path.exists(cred_path):
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                'projectId': FIREBASE_PROJECT_ID
            })
            print(f"Firebase initialized with serviceAccountKey.json from {cred_path}")
        except Exception as e:
            print(f"Failed to initialize with serviceAccountKey.json: {str(e)}")
            # Only try Application Default Credentials as a fallback
            try:
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {
                    'projectId': FIREBASE_PROJECT_ID
                })
                print("Firebase initialized with Application Default Credentials")
            except Exception as e:
                print(f"Failed to initialize Firebase: {str(e)}")
    else:
        print(f"No serviceAccountKey.json found at {cred_path}")
        raise Exception("Firebase credentials not found")


class FirebaseAuthentication(authentication.BaseAuthentication):
    def authenticate(self, request):
        """
        Authenticates the request using a Firebase ID token.
        """
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        print(f"Auth header received: {auth_header}")  # Debug log
        print(f"Request headers: {[f'{k}: {v}' for k, v in request.META.items() if k.startswith('HTTP_')]}")

        if not auth_header:
            print("No auth header provided - checking for OPTIONS request")  # Debug log
            # For OPTIONS requests (CORS preflight), we'll return None to allow the request
            if request.method == 'OPTIONS':
                print("OPTIONS request detected - skipping auth")
                return None
            return None  # No authentication provided

        try:
            # Check for Bearer token
            if not auth_header.startswith('Bearer '):
                print("Token does not have Bearer prefix")  # Debug log
                return None

            # Extract the token
            token = auth_header.split('Bearer ')[1].strip()

            # More verbose logging for troubleshooting
            print(f"Token extracted, length: {len(token)}")
            print(f"Token first 10 chars: {token[:10]}...")
            print(f"Token format check - contains dots: {('.' in token)}")
            print(f"Request path: {request.path}")
            print(f"Request method: {request.method}")

            # Check if token is being sent as a URL path
            if request.path.startswith('/' + token):
                print("ERROR: Token is being sent as URL path instead of Authorization header")
                raise exceptions.AuthenticationFailed('Token should be sent in Authorization header, not as URL path')

        except (ValueError, IndexError) as e:
            print(f"Error extracting token: {str(e)}")  # Debug log
            raise exceptions.AuthenticationFailed('Invalid token header format.')

        if not token:
            print("Empty token received")  # Debug log
            return None  # No token provided

        try:
            print("Attempting to verify Firebase token...")  # Debug log

            # Remove the development bypass and properly verify the token
            try:
                decoded_token = auth.verify_id_token(token)  # Verify the token
                uid = decoded_token['uid']  # Get the user ID from the token
                print(f"Token verified successfully for user: {uid}")  # Debug log
            except Exception as token_error:
                print(f"Token verification failed: {str(token_error)}")
                # Don't bypass security, but provide detailed error for debugging
                raise exceptions.AuthenticationFailed(f'Firebase token verification failed: {str(token_error)}')

        except auth.InvalidIdTokenError as e:
            print(f"Invalid Firebase ID token: {str(e)}")  # Debug log
            raise exceptions.AuthenticationFailed('Invalid Firebase ID token: ' + str(e))
        except ValueError as e:
            print(f"Invalid token format: {str(e)}")  # Debug log
            raise exceptions.AuthenticationFailed('Invalid Firebase ID token format' + str(e))
        except Exception as e:
            print(f"Unexpected error during authentication: {str(e)}")  # Debug log
            raise exceptions.AuthenticationFailed('Firebase authentication failed: '+ str(e))

        # Get or create the Django user.
        try:
            user = User.objects.get(username=uid)
            print(f"Found existing user: {user.username}")
        except User.DoesNotExist:
            # Create a new user.  You might want to customize this.
            print(f"Creating new user with username: {uid}")
            try:
                user = User.objects.create_user(
                    username=uid,
                    email=f"{uid}@example.com",
                    password="devpassword123"  # Only for development!
                )
                print(f"New user created: {user.username}")
            except Exception as e:
                print(f"Error creating user: {str(e)}")
                # If user creation fails, try to get the user again
                user = User.objects.get(username=uid)
                print(f"Retrieved existing user after creation error: {user.username}")

        # Add Django user ID to the request for use in views
        request.user_id = user.id
        return (user, None) # Return user and set authentication to True