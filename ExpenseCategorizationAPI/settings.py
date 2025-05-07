import os
from pathlib import Path # Ensure Path is imported if you use it elsewhere
import firebase_admin # For Firebase Admin SDK initialization

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Environment Variables (loaded via load_params.py from SSM) ---
# These should be set in your environment by the load_params.py script
# or via GitHub secrets in the CI/CD pipeline for steps like collectstatic.

AWS_STORAGE_BUCKET_NAME = os.environ.get('AWS_STORAGE_BUCKET_NAME')
AWS_S3_REGION_NAME = os.environ.get('AWS_S3_REGION_NAME', 'af-south-1') # Default if not set
CLOUDFRONT_DOMAIN = os.environ.get('CLOUDFRONT_DOMAIN')
SECRET_KEY = os.environ.get('SECRET_KEY')
DEBUG_ENV = os.environ.get('DEBUG', 'False').lower()
DEBUG = DEBUG_ENV == 'true'

# Ensure SECRET_KEY is set in production (when DEBUG is False)
if not DEBUG and not SECRET_KEY:
    raise ValueError("CRITICAL: SECRET_KEY must be set in the environment for production!")

# ALLOWED_HOSTS should also be loaded from environment
ALLOWED_HOSTS_STRING = os.environ.get('ALLOWED_HOSTS', '')
ALLOWED_HOSTS = [host.strip() for host in ALLOWED_HOSTS_STRING.split(',') if host.strip()]
if not DEBUG and not ALLOWED_HOSTS:
    # In a real production scenario, you might want to raise an error or have a restrictive default.
    # For now, if it's not set and not DEBUG, it will likely fail later, which is fine for alerting.
    print("WARNING: ALLOWED_HOSTS is not set in the environment for production!")


# --- Static Files Configuration (S3 and CloudFront) ---

# The top-level directory (prefix) in your S3 bucket for static files.
# Example: if 'static', files will be at s3://your-bucket/static/admin/css/base.css
AWS_LOCATION = 'static'

if AWS_STORAGE_BUCKET_NAME and CLOUDFRONT_DOMAIN:
    # Static files storage backend for S3 (standard from django-storages)
    STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

    # The URL CloudFront will use to serve static files.
    # It should match the S3 structure under AWS_LOCATION.
    STATIC_URL = f'https://{CLOUDFRONT_DOMAIN}/{AWS_LOCATION}/'

    # This tells django-storages to use the CloudFront domain for generating static file URLs
    # AWS_S3_CUSTOM_DOMAIN = CLOUDFRONT_DOMAIN

    # Optional: S3 object parameters like Cache-Control
    AWS_S3_OBJECT_PARAMETERS = {
        'CacheControl': 'max-age=86400', # Cache for 1 day (adjust as needed)
    }
    # Ensure bucket is private and access is via CloudFront OAC
    AWS_DEFAULT_ACL = None
    AWS_S3_FILE_OVERWRITE = True # Default is True, but explicit can be good
    AWS_QUERYSTRING_AUTH = False # Do not use querystring auth for static files via CloudFront

    print(f"INFO: Production static files configured. STATIC_URL: {STATIC_URL}")

else:
    # Fallback for local development if S3/CloudFront variables are not set
    # Or if running manage.py commands locally without full S3 setup.
    print("INFO: S3/CloudFront environment variables not fully set. Using local static files configuration.")
    STATIC_URL = '/static/'
    # For local `collectstatic` if you want to test collection to a local dir:
    STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_collected_local')
    # For local development, you might use the default staticfiles storage:
    # STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.StaticFilesStorage'

# STATIC_ROOT: This is where `collectstatic` will gather files from your apps locally
# *before* a storage backend (like S3Boto3Storage) processes them if it needs a local cache.
# The S3Boto3Storage backend typically uploads files found by Django's
# staticfiles finders directly to S3 under AWS_LOCATION. The name of this STATIC_ROOT
# directory itself does not become part of the S3 path when using standard
# S3Boto3Storage with AWS_LOCATION. It's primarily for the finders.
# If STATICFILES_STORAGE is the default local storage, then collectstatic
# would put files into this STATIC_ROOT directory.
# This value is fine for the GitHub Actions runner.
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles_local_temp_collection')


# --- Other Settings (Ensure these are consistent with your project) ---

# Updated Login URL (obscured admin path)
LOGIN_URL = '/manage-site-fintrackke/' # Or whatever you set in your main urls.py

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles', # Manages static files
    'storages',                 # For django-storages (S3)
    'rest_framework',
    'expenses.apps.ExpensesConfig',
    'investment',
    'corsheaders',
    'drf_yasg',
    'ai_service',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ExpenseCategorizationAPI.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # Add if you have project-level templates
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ExpenseCategorizationAPI.wsgi.application'


# Database (ensure your load_params.py sets these env vars from SSM)
DATABASES = {
    'default': {
        'ENGINE': os.environ.get('DB_ENGINE'),
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT'),
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC' # Standard for servers
USE_I18N = True
USE_TZ = True # Recommended for handling timezones correctly

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Django REST Framework settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': ['expenses.firebase_auth.FirebaseAuthentication'],
    'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.IsAuthenticated'],
    'EXCEPTION_HANDLER': 'rest_framework.views.exception_handler',
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
        # Remove BrowsableAPIRenderer for production if API browser not needed/wanted
        'rest_framework.renderers.BrowsableAPIRenderer' if DEBUG else 'rest_framework.renderers.JSONRenderer',
    ),
    'DEFAULT_SCHEMA_CLASS': 'rest_framework.schemas.coreapi.AutoSchema', # Consider drf-spectacular for OpenAPI 3
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'DEFAULT_VERSION': 'v1',
    'ALLOWED_VERSIONS': ['v1', 'v2'],
    'VERSION_PARAM': 'version',
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# CORS settings (Ensure CORS_ALLOWED_ORIGINS is set in your SSM parameters for production)
default_cors_origins_dev = [ # For local development only
    'http://localhost:19006', 'http://localhost:8081', 'http://localhost:8080',
    'http://localhost:19000', 'http://localhost:3000', 'http://127.0.0.1:19006',
    'http://127.0.0.1:8080', 'http://127.0.0.1:8000', 'http://127.0.0.1:19000',
]
cors_origins_env = os.environ.get('CORS_ALLOWED_ORIGINS', '')
if cors_origins_env:
    CORS_ALLOWED_ORIGINS = [origin.strip() for origin in cors_origins_env.split(',') if origin.strip()]
elif DEBUG: # Only use default dev origins if DEBUG is True and no env var is set
    CORS_ALLOWED_ORIGINS = default_cors_origins_dev
else: # For production, if not set in env, block all or set a restrictive default
    CORS_ALLOWED_ORIGINS = [] # Or specific production frontend URL(s) from SSM

CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ['DELETE', 'GET', 'OPTIONS', 'PATCH', 'POST', 'PUT']
CORS_ALLOW_HEADERS = [
    'accept', 'accept-encoding', 'authorization', 'content-type', 'dnt', 'origin',
    'user-agent', 'x-csrftoken', 'x-requested-with', 'x-debug-mode', 'x-client-platform',
]
CORS_EXPOSE_HEADERS = ['Content-Type', 'X-CSRFToken']
# CSRF_TRUSTED_ORIGINS should ideally be a more specific list for production,
# matching your frontend domain(s) served over HTTPS.
# Using CORS_ALLOWED_ORIGINS here is a common convenience but review for production.
CSRF_TRUSTED_ORIGINS = CORS_ALLOWED_ORIGINS


# HTTPS Security Settings
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = not DEBUG # Only redirect if not in DEBUG mode
SESSION_COOKIE_SECURE = not DEBUG # Only secure if not in DEBUG mode
CSRF_COOKIE_SECURE = not DEBUG # Only secure if not in DEBUG mode
CSRF_COOKIE_HTTPONLY = True # Added as per our discussion
SECURE_HSTS_SECONDS = 0 if DEBUG else 3600 # Start with 1 hour for HSTS, increase to 31536000 (1 year) once confident. 0 in DEBUG.
SECURE_HSTS_INCLUDE_SUBDOMAINS = False # Set to True only if all subdomains are HTTPS
SECURE_HSTS_PRELOAD = False # Set to True only after meeting preload requirements
X_FRAME_OPTIONS = 'DENY' # Explicitly set, although it's Django's default
SECURE_CONTENT_TYPE_NOSNIFF = True # Explicitly set, although it's Django's default with SecurityMiddleware

DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB

# Celery Configuration
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') # e.g., 'sqs://'
CELERY_TASK_CREATE_MISSING_QUEUES = False
CELERY_BROKER_TRANSPORT_OPTIONS = {
    'region': os.environ.get('AWS_S3_REGION_NAME', 'af-south-1'), # Use same region as S3 or specific SQS region
    'visibility_timeout': 3600, # Default SQS visibility timeout
    'polling_interval': 1, # How often to poll (seconds)
    'predefined_queues': {
        'celery': { # Default Celery queue name
            'url': os.environ.get('CELERY_SQS_QUEUE_URL'), # Full SQS Queue URL from SSM
        }
    }
}
CELERY_ACCEPT_CONTENT = ['application/json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json' # Though RESULT_BACKEND is None, good practice
CELERY_TIMEZONE = os.environ.get('CELERY_TIMEZONE', 'Africa/Nairobi') # Or your server's timezone
CELERY_RESULT_BACKEND = None # No results backend configured for simplicity
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_SEND_SENT_EVENT = True # If you have event consumers
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True

# Firebase Initialization (ensure serviceAccountKey.json is handled securely)
# It's better to load from an environment variable or secure file path not in repo
# The path below assumes serviceAccountKey.json is in the project's root (BASE_DIR)
# For production, FIREBASE_SERVICE_ACCOUNT_PATH should be set in the environment
# by load_params.py to an absolute path on the server where the key file is securely stored.
FIREBASE_SERVICE_ACCOUNT_PATH = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH', os.path.join(BASE_DIR, 'serviceAccountKey.json'))
try:
    if os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
        if not firebase_admin._apps: # Initialize only if no apps are present
            firebase_admin.initialize_app(firebase_admin.credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH))
            print("INFO: Firebase Admin SDK initialized successfully.")
        else:
            print("INFO: Firebase Admin SDK already initialized.")
    else:
        print(f"WARNING: Firebase service account key file not found at path: {FIREBASE_SERVICE_ACCOUNT_PATH}. Firebase Admin SDK not initialized.")
except Exception as e:
    print(f"ERROR: Failed to initialize Firebase Admin SDK: {e}")


# AI Service Settings
GOOGLE_AI_STUDIO_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-1.0-pro') # Or your preferred model

# AI Service Feature Flags
ENABLE_GEMINI_AI = os.environ.get('ENABLE_GEMINI_AI', 'true').lower() == 'true'
ENABLE_EXPENSE_AI = os.environ.get('ENABLE_EXPENSE_AI', 'true').lower() == 'true'
ENABLE_INVESTMENT_AI = os.environ.get('ENABLE_INVESTMENT_AI', 'true').lower() == 'true'

# Logging Configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple' if DEBUG else 'verbose', # Simple in DEBUG, verbose in PROD
        },
        # Add other handlers like file handlers or CloudWatch logs handler if needed for production
    },
    'root': { # Configure root logger to see logs from all apps
        'handlers': ['console'],
        'level': 'INFO', # Default root level
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False, # Don't pass to root logger if handled here
        },
        'django.request': { # More detailed request/response logging
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'WARNING', # DEBUG in dev, WARNING in prod
            'propagate': False,
        },
        # Configure logging for your specific apps if needed
        'expenses': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
        'investment': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
        'ai_service': {
            'handlers': ['console'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
    },
}

# Print statements for verifying settings during startup (optional, remove for cleaner prod logs)
if DEBUG:
    print(f"DEBUG mode is ON. Allowed hosts: {ALLOWED_HOSTS}")
    print(f"CORS Origins: {CORS_ALLOWED_ORIGINS}")
else:
    print(f"DEBUG mode is OFF. Allowed hosts: {ALLOWED_HOSTS}")
    print(f"CORS Origins: {CORS_ALLOWED_ORIGINS}")
    if not CORS_ALLOWED_ORIGINS:
        print("WARNING: CORS_ALLOWED_ORIGINS is empty in production!")

# Ensure all environment variables are loaded by load_params.py before Django starts
# This settings file assumes those variables are already in os.environ
