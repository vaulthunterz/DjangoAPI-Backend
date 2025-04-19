# API Documentation

## 1. Project Overview

### Purpose and Scope
This project is a comprehensive financial management system that combines expense tracking with investment recommendations. The application provides automated expense categorization using machine learning, along with personalized investment advice based on user spending patterns and financial goals.

Key capabilities include:
- Automated expense categorization and tracking
- Investment portfolio management
- AI-powered financial recommendations
- Natural language processing for expense descriptions
- Data visualization for financial insights

The system serves as a backend API for financial management applications, providing a robust set of endpoints that can be consumed by various frontend clients.

### Technologies Used
- **Django & Django REST Framework**: Core backend framework and REST API implementation
- **PostgreSQL**: Primary database (implied from Django ORM usage)
- **Firebase**: Authentication and user management
- **Machine Learning**: Custom ML models for expense categorization
- **Google Gemini AI**: Integration for natural language processing capabilities
- **Swagger/OpenAPI**: API documentation and testing
- **Docker**: Containerization for deployment

Key dependencies include REST framework extensions, CORS handling, and various AI/ML libraries.

### System Architecture
The system follows a modular architecture with several key components:

1. **Core API Layer**: Django REST Framework providing the API endpoints
2. **Application Services**:
   - Expenses Service: Managing transaction data and categorization
   - Investment Service: Portfolio management and recommendations
   - AI Service: ML model integration and prediction services

3. **Data Layer**: Data persistence through Django ORM to PostgreSQL
4. **External Integrations**: Firebase for authentication, Google Gemini for AI capabilities

Authentication is handled through Firebase, with all API endpoints requiring authenticated access (except documentation endpoints).

The system implements a clean separation of concerns:
- Models define the data structure
- Serializers handle data validation and transformation
- Views implement the API endpoints and business logic
- AI services provide prediction and recommendation capabilities

Data flows through the system as follows:
1. Client request is authenticated via Firebase tokens
2. Appropriate service handles the request
3. For transactions/investments requiring categorization or recommendations, AI services are invoked
4. Results are returned via RESTful JSON responses

This architecture allows for scalable and maintainable development, with clear separation between different functional areas of the application.

## 2. Project Structure

### 2.1 Directory Organization

The project follows a standard Django application structure with specialized components for AI/ML functionality. Below is the comprehensive directory structure:

```
DjangoAPI-Backend/
├── ExpenseCategorizationAPI/    # Django project configuration
│   ├── __init__.py              # Package initialization with Django app config
│   ├── settings.py              # Project settings and configuration
│   ├── urls.py                  # Main URL routing configuration
│   ├── wsgi.py                  # WSGI application configuration for deployment
│   ├── asgi.py                  # ASGI application configuration for async deployment
│   └── celery.py                # Celery configuration for asynchronous tasks
├── expenses/                    # Expense tracking application
│   ├── __init__.py              # Package initialization
│   ├── admin.py                 # Django admin configuration (empty)
│   ├── apps.py                  # Django app configuration
│   ├── models.py                # Database models for expenses
│   ├── views.py                 # API views for expenses
│   ├── serializers.py           # Serializers for expense models
│   ├── urls.py                  # URL routing for expense endpoints
│   ├── pagination.py            # Custom pagination for API responses
│   ├── firebase_auth.py         # Firebase authentication configuration
│   ├── ml/                      # Machine learning models for expense categorization
│   │   ├── __init__.py          # Package initialization for ML module
│   │   ├── model_training.py    # Machine learning model training logic
│   │   ├── predictor.py         # Prediction service implementation
│   │   ├── train_model.py       # Model training script
│   │   ├── utils.py             # Utility functions for ML models
│   │   ├── custom_model.joblib  # Custom trained ML model
│   │   ├── vectorizer.joblib    # Text vectorizer for ML models
│   │   ├── label_encoder.joblib # Label encoder for categories
│   │   ├── trained_models/      # Saved trained ML models
│   │   │   ├── rf_classifier.joblib     # Random Forest classifier model
│   │   │   ├── expense_classifier.joblib # Expense categorization model
│   │   │   └── label_encoders.joblib    # Label encoders for categories
│   │   └── training_data/       # Training data for ML models
│   │       └── transactions.csv # Sample transaction data for training
│   ├── management/              # Django management commands
│   │   ├── __init__.py          # Package initialization
│   │   └── commands/            # Custom management commands
│   │       ├── __init__.py      # Package initialization
│   │       ├── create_initial_data.py  # Command to create initial data
│   │       └── populate_categories.py  # Command to populate expense categories
│   └── migrations/              # Database migrations
│       └── __init__.py          # Package initialization for migrations
├── investment/                  # Investment recommendation application
│   ├── __init__.py              # Package initialization
│   ├── admin.py                 # Django admin configuration
│   ├── apps.py                  # Django app configuration
│   ├── models.py                # Database models for investments
│   ├── views.py                 # API views for investments
│   ├── serializers.py           # Serializers for investment models
│   ├── urls.py                  # URL routing for investment endpoints
│   ├── pagination.py            # Custom pagination for API responses
│   ├── services.py              # Business logic for investment operations
│   ├── tasks.py                 # Asynchronous tasks for the investment app
│   ├── tests.py                 # Basic tests for the investment app
│   ├── ml_recommender.py        # ML-based recommendation engine
│   ├── ml/                      # ML models for investment recommendations
│   │   ├── __init__.py          # Package initialization for ML module
│   │   ├── recommender/         # Recommendation algorithms
│   │   │   ├── base.py          # Base recommender class
│   │   │   ├── random_forest.py # Random Forest based recommender
│   │   │   ├── hybrid_recommender.py # Hybrid recommendation system
│   │   │   └── advanced_hybrid_recommender.py # Advanced hybrid recommender
│   │   └── utils/               # Utility functions for ML models
│   │       └── preprocessing.py # Data preprocessing utilities
│   └── migrations/              # Database migrations
│       ├── __init__.py          # Package initialization for migrations
│       └── 0001_initial.py      # Initial database migration
├── ai_service/                  # AI service application
│   ├── tests/                   # Unit tests for AI services
│   │   ├── __init__.py          # Package initialization
│   │   ├── README.md            # Testing documentation
│   │   ├── run_detailed_tests.py # Script for running detailed tests
│   │   ├── run_tests.py         # Script for running basic tests
│   │   ├── test_runner.py       # Custom test runner
│   │   ├── test_simple.py       # Simple test cases
│   │   ├── test_django.py       # Django integration tests
│   │   ├── test_service.py      # Base service tests
│   │   ├── test_factory.py      # Factory tests
│   │   ├── test_expense_service.py # Expense service tests
│   │   ├── test_investment_service.py # Investment service tests
│   │   └── test_gemini_service.py # Gemini service tests
│   ├── __init__.py              # Package initialization
│   ├── apps.py                  # Django app configuration
│   ├── expense_service.py       # Expense prediction service
│   ├── investment_service.py    # Investment recommendation service
│   ├── gemini_service.py        # Google Gemini AI integration
│   ├── factory.py               # Factory for creating AI services
│   ├── service.py               # Base service class
│   ├── settings.py              # AI service specific settings
│   ├── views.py                 # API views for AI services
│   ├── urls.py                  # URL routing for AI service endpoints
│   └── README.md                # Documentation for AI service
├── staticfiles/                 # Collected static files for production
│   ├── admin/                   # Django admin interface static files
│   │   ├── css/                 # Admin CSS styles
│   │   │   ├── base.css         # Base admin styles
│   │   │   ├── forms.css        # Form styling
│   │   │   ├── login.css        # Login page styling
│   │   │   ├── responsive.css   # Responsive design styles
│   │   │   ├── widgets.css      # Form widget styles
│   │   │   └── vendor/          # Third-party CSS
│   │   │       └── select2/     # Select2 dropdown library
│   │   ├── img/                 # Admin interface images
│   │   │   ├── gis/             # GIS-related icons
│   │   │   ├── icon-*.svg       # Various interface icons
│   │   │   └── sorting-icons.svg # Table sorting icons
│   │   └── js/                  # Admin JavaScript
│   │       ├── admin/           # Admin-specific scripts
│   │       │   ├── DateTimeShortcuts.js # Date/time widgets
│   │       │   └── RelatedObjectLookups.js # Related field lookups
│   │       ├── actions.js       # Bulk action handling
│   │       ├── core.js          # Core functionality
│   │       └── vendor/          # Third-party JavaScript
│   │           ├── jquery/      # jQuery library
│   │           ├── select2/     # Select2 dropdown library
│   │           └── xregexp/     # Extended regular expressions
│   └── rest_framework/          # Django REST Framework static files
│       ├── css/                 # DRF CSS styles
│       ├── docs/                # API documentation styles
│       │   ├── css/             # Documentation CSS
│       │   ├── img/             # Documentation images
│       │   └── js/              # Documentation JavaScript
│       ├── fonts/               # Web fonts
│       ├── img/                 # DRF interface images
│       └── js/                  # DRF JavaScript files
├── .idea/                       # IDE configuration (PyCharm)
│   ├── inspectionProfiles/      # Code inspection profiles
│   │   ├── Project_Default.xml  # Default inspection profile
│   │   └── profiles_settings.xml # Inspection profile settings
│   ├── .gitignore               # Git ignore rules for IDE files
│   ├── ExpenseCategorizationAPI.iml # IntelliJ module file
│   ├── ShortcutLearner.xml      # Shortcut configuration
│   ├── dataSources.xml          # Database connection settings
│   ├── encodings.xml            # File encodings configuration
│   ├── misc.xml                 # Miscellaneous settings
│   ├── modules.xml              # Project modules configuration
│   ├── sqldialects.xml          # SQL dialect settings
│   └── vcs.xml                  # Version control settings
├── manage.py                    # Django management script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── .dockerignore                # Docker ignore rules
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
├── serviceAccountKey.json       # Firebase service account key
├── README.md                    # Project README
└── API_DOCUMENTATION.md         # This API documentation file
```

### 2.2 Key Components

#### 2.2.1 Core Project Files

- **ExpenseCategorizationAPI/**: The main Django project configuration.
  - `settings.py`: Contains all the Django settings including database configuration, middleware, installed apps, etc.
  - `urls.py`: Defines the main URL routing for the application.
  - `wsgi.py` & `asgi.py`: Entry points for WSGI/ASGI servers.

- **manage.py**: Django's command-line utility for administrative tasks.
- **requirements.txt**: Lists all Python package dependencies.
- **.env**: Contains environment variables for configuration.
- **Dockerfile**: Configuration for Docker containerization.

#### 2.2.2 Application Modules

**Expenses App**
- Handles all expense tracking and categorization functionality.
- `models.py`: Defines database schema for transactions, categories, and subcategories.
- `views.py`: Contains API endpoints for transaction CRUD operations.
- `ml/`: Houses machine learning models for automated expense categorization.
- `firebase_auth.py`: Custom authentication using Firebase.

**Investment App**
- Manages investment portfolios and recommendations.
- `models.py`: Defines investment portfolios, assets, and user profiles.
- `views.py`: Implements API endpoints for portfolio management.
- `ml/recommender/`: Contains recommendation algorithms for investments.
- `services.py`: Business logic for investment operations.

**AI Service App**
- Provides AI functionality to other applications.
- `expense_service.py`: Service for expense categorization using ML.
- `investment_service.py`: Service for investment recommendations.
- `gemini_service.py`: Integration with Google Gemini for natural language processing.
- `factory.py`: Factory pattern implementation for creating AI service instances.

### 2.3 Configuration Files

- **settings.py**: Main Django settings file containing database configuration, installed apps, middleware, static files configuration, etc.
- **.env**: Environment variables including:
  - Database connection parameters
  - API keys and secrets
  - Firebase configuration
  - Debug settings
  - Allowed hosts
  
- **serviceAccountKey.json**: Firebase authentication configuration.
- **Dockerfile**: Defines how the application is containerized for deployment.
- **.dockerignore**: Specifies files to exclude from Docker containers.

### 2.4 Database Schema

The database is designed with clear relationships between the main entities:

1. **Expenses**:
   - Transactions
   - Categories
   - Subcategories

2. **Investments**:
   - User Profiles
   - Portfolios
   - Portfolio Items
   - Assets

The schema is normalized to minimize redundancy while maintaining referential integrity between related entities. 

## 3. Main Applications

The backend consists of three main Django applications that work together to provide the complete financial management functionality. Each application has a specific focus but integrates with the others to create a cohesive system.

### 3.1 Expenses App

#### 3.1.1 Overview
The Expenses app is the core module responsible for tracking and categorizing financial transactions. It provides API endpoints for managing transactions, categories, and subcategories.

#### 3.1.2 Key Features
- Transaction management (create, read, update, delete)
- Automated categorization of transactions using machine learning
- Category and subcategory organization
- Firebase authentication integration
- Pagination for optimized data fetching

#### 3.1.3 Models
- **Transaction**: Represents financial transactions with properties such as amount, description, merchant, time, and category.
- **Category**: Top-level categories for transactions (e.g., Food, Transportation, Entertainment).
- **SubCategory**: More specific categorization under parent categories (e.g., Restaurants under Food).

#### 3.1.4 API Endpoints
- `/api/expenses/transactions/`: CRUD operations for transactions
- `/api/expenses/categories/`: Manage expense categories
- `/api/expenses/subcategories/`: Manage expense subcategories
- `/api/expenses/category-lookup/`: Quick lookup for categories
- `/api/expenses/subcategory-lookup/`: Quick lookup for subcategories
- `/api/expenses/change-password/`: Change user password

#### 3.1.5 ML Integration
The Expenses app includes a machine learning component that automatically categorizes transactions based on their description and merchant information. This is implemented in the `ml/` directory, which contains:
- Pre-trained models for expense categorization
- Vectorizers for text processing
- Training scripts for improving model accuracy

### 3.2 Investment App

#### 3.2.1 Overview
The Investment app handles portfolio management and provides investment recommendations. It tracks investment assets, analyzes portfolios, and suggests investment strategies based on user profiles.

#### 3.2.2 Key Features
- Portfolio creation and management
- Asset tracking and valuation
- Investment performance monitoring
- Returns calculation
- Portfolio analysis and asset allocation
- Investment recommendations

#### 3.2.3 Models
- **UserProfile**: Extended user information for investment preferences and risk tolerance.
- **Portfolio**: Collection of investment assets belonging to a user.
- **PortfolioItem**: Individual assets within a portfolio, including quantity and purchase information.
- **Asset**: Investment products available for portfolios.

#### 3.2.4 API Endpoints
- `/api/investment/portfolios/`: CRUD operations for investment portfolios
- `/api/investment/portfolio-items/`: Manage items within portfolios
- `/api/investment/assets/`: Available investment assets
- `/api/investment/summary/`: Summary of all user portfolios
- `/api/investment/performance/`: Portfolio performance metrics

#### 3.2.5 ML Recommender
The Investment app includes a machine learning recommendation engine that suggests investments based on:
- User profile and preferences
- Risk tolerance
- Expense patterns
- Market trends
- Current portfolio composition

The recommendation system uses hybrid approaches combining collaborative filtering, content-based filtering, and rule-based systems.

### 3.3 AI Service App

#### 3.3.1 Overview
The AI Service app acts as a bridge between the machine learning models and the API endpoints. It provides services for expense categorization, investment recommendations, and natural language processing through Google Gemini integration.

#### 3.3.2 Key Features
- Expense prediction and categorization
- Investment recommendation generation
- Portfolio analysis
- Expense-based investment suggestions
- Gemini AI integration for natural language understanding
- Service factory pattern for flexible AI implementation

#### 3.3.3 Services
- **ExpenseService**: Handles expense categorization and prediction
- **InvestmentService**: Provides investment recommendations
- **GeminiService**: Integrates with Google Gemini for advanced NLP capabilities

#### 3.3.4 API Endpoints
- `/api/ai/expense/predict/`: Predict category for an expense
- `/api/ai/expense/predict/custom/`: Custom prediction with specific parameters
- `/api/ai/expense/metrics/`: Get expense model metrics
- `/api/ai/expense/train/`: Trigger training of the expense model
- `/api/ai/gemini/predict/`: Make predictions using Google Gemini
- `/api/ai/gemini/chat/`: Chat interface with Gemini AI
- `/api/ai/investment/predict/`: Generate investment recommendations
- `/api/ai/investment/analyze/<portfolio_id>/`: Analyze a specific portfolio
- `/api/ai/investment/expense-recommendations/<user_id>/`: Get investment recommendations based on expense history
- `/api/ai/info/`: Get information about available AI services

#### 3.3.5 Factory Pattern
The AI Service app implements the factory pattern to create instances of different AI services. This design allows for:
- Consistent interface across different AI implementations
- Easy swapping of underlying AI models
- Centralized configuration for AI services
- Testing with mock implementations 

### 3.4 Implementation Details

#### 3.4.1 Authentication Flow

The application uses Firebase Authentication for user management. Here's how authentication works:

1. **User Registration/Login**: The frontend handles user registration and login through Firebase Authentication.

2. **Token Acquisition**: Upon successful authentication, Firebase provides an ID token.

3. **Token Usage**: For authenticated API requests, the frontend must include the token in the `Authorization` header:
   ```
   Authorization: Bearer <firebase_id_token>
   ```

4. **Token Validation**: The backend validates the token using the `firebase_auth.py` module in the expenses app:
   - Verifies the token is properly signed
   - Confirms the token is not expired
   - Extracts the user identifier

5. **Permission Checks**: After authentication, permission checks are applied based on the `IsAuthenticated` permission class.

Example authentication flow:
```python
# Backend token verification (simplified)
def verify_firebase_token(request):
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    if not auth_header.startswith('Bearer '):
        return None
    
    token = auth_header.split(' ')[1]
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        return uid
    except Exception:
        return None
```

#### 3.4.2 Request/Response Examples

Here are examples of request and response payloads for key endpoints:

**Creating a Transaction**

Request:
```json
POST /api/expenses/transactions/
{
  "description": "Grocery shopping at Whole Foods",
  "amount": 85.42,
  "merchant": "Whole Foods",
  "transaction_type": "expense",
  "time_of_transaction": "2023-04-15T14:30:00Z",
  "category": 3,
  "subcategory": 12
}
```

Response:
```json
{
  "id": 243,
  "description": "Grocery shopping at Whole Foods",
  "amount": "85.42",
  "merchant": "Whole Foods",
  "transaction_type": "expense",
  "time_of_transaction": "2023-04-15T14:30:00Z",
  "category": 3,
  "subcategory": 12,
  "created_at": "2023-04-15T14:35:22Z",
  "updated_at": "2023-04-15T14:35:22Z"
}
```

**Creating a Portfolio**

Request:
```json
POST /api/investment/portfolios/
{
  "name": "Retirement Fund",
  "description": "Long-term retirement investments",
  "risk_level": "moderate",
  "investment_strategy": "balanced_growth"
}
```

Response:
```json
{
  "id": 16,
  "name": "Retirement Fund",
  "description": "Long-term retirement investments",
  "risk_level": "moderate",
  "investment_strategy": "balanced_growth",
  "total_amount": "0.00",
  "created_at": "2023-04-16T09:22:15Z",
  "updated_at": "2023-04-16T09:22:15Z"
}
```

**AI Expense Prediction**

Request:
```json
POST /api/ai/expense/predict/
{
  "description": "Uber ride to airport",
  "merchant": "Uber"
}
```

Response:
```json
{
  "predicted_category": {
    "id": 5,
    "name": "Transportation"
  },
  "predicted_subcategory": {
    "id": 18,
    "name": "Ride Sharing"
  },
  "confidence": 0.92,
  "alternative_categories": [
    {
      "category": {
        "id": 8,
        "name": "Travel"
      },
      "confidence": 0.07
    }
  ]
}
```

#### 3.4.3 Model Relationships

The database schema includes several important relationships between models:

**Expenses App**:
1. **Transaction to Category**: Many-to-one relationship
   - Each transaction belongs to one category
   - A category can have multiple transactions
   - Foreign key: `Transaction.category → Category.id`

2. **Transaction to SubCategory**: Many-to-one relationship
   - Each transaction can have one subcategory
   - A subcategory can be used by multiple transactions
   - Foreign key: `Transaction.subcategory → SubCategory.id`

3. **SubCategory to Category**: Many-to-one relationship
   - Each subcategory belongs to one parent category
   - A category can have multiple subcategories
   - Foreign key: `SubCategory.category → Category.id`

**Investment App**:
1. **Portfolio to UserProfile**: Many-to-one relationship
   - Each portfolio belongs to one user profile
   - A user can have multiple portfolios
   - Foreign key: `Portfolio.user → UserProfile.id`

2. **PortfolioItem to Portfolio**: Many-to-one relationship
   - Each portfolio item belongs to one portfolio
   - A portfolio can contain multiple items
   - Foreign key: `PortfolioItem.portfolio → Portfolio.id`

3. **PortfolioItem to Asset**: Many-to-one relationship
   - Each portfolio item references one asset type
   - An asset can be included in multiple portfolios
   - Foreign key: `PortfolioItem.asset → Asset.id`

ER Diagram (simplified text representation):
```
User Profile 1 --- * Portfolio 1 --- * Portfolio Item * --- 1 Asset
Transaction * --- 1 Category 1 --- * SubCategory
```

#### 3.4.4 Error Handling

The API uses standard HTTP status codes and structured JSON responses for error handling:

**Common Status Codes**:
- `200 OK`: Successful request
- `201 Created`: Resource successfully created
- `400 Bad Request`: Invalid input or parameters
- `401 Unauthorized`: Authentication required or failed
- `403 Forbidden`: Authenticated but insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Unexpected server error

**Error Response Format**:
```json
{
  "error": "Brief error message",
  "detail": "More detailed explanation",
  "code": "ERROR_CODE_IDENTIFIER"
}
```

**Common Error Scenarios**:
1. **Authentication Errors**:
   ```json
   {
     "error": "Authentication failed",
     "detail": "Invalid or expired token",
     "code": "AUTH_FAILED"
   }
   ```

2. **Validation Errors**:
   ```json
   {
     "error": "Validation failed",
     "detail": {
       "amount": ["This field is required."],
       "category": ["Invalid category ID."]
     },
     "code": "VALIDATION_ERROR"
   }
   ```

3. **Resource Not Found**:
   ```json
   {
     "error": "Not found",
     "detail": "Transaction with ID 123 does not exist",
     "code": "NOT_FOUND"
   }
   ```

#### 3.4.5 Filtering & Pagination

The API supports filtering, sorting, and pagination for list endpoints using query parameters:

**Pagination**:
- All list endpoints support pagination
- Default page size varies by endpoint (typically 10-20 items)
- Example: `/api/expenses/transactions/?page=2&page_size=15`

**Response Format for Paginated Results**:
```json
{
  "count": 243,
  "next": "http://api.example.com/api/expenses/transactions/?page=3",
  "previous": null,
  "results": [
    { /* item 1 */ },
    { /* item 2 */ },
    // ...more items
  ]
}
```

**Filtering**:
Different endpoints support specific filters:

1. **Transactions**:
   - By date range: `?start_date=2023-01-01&end_date=2023-01-31`
   - By category: `?category=5`
   - By transaction type: `?transaction_type=expense`
   - By amount range: `?min_amount=10.00&max_amount=100.00`
   - By merchant: `?merchant=Starbucks`
   - Text search: `?search=grocery`

2. **Portfolios**:
   - By risk level: `?risk_level=moderate`
   - By strategy: `?investment_strategy=growth`
   - By created date: `?created_after=2023-01-01`

3. **Assets**:
   - By asset type: `?asset_type=stock`
   - By risk level: `?risk_level=high`

**Sorting**:
- Use the `ordering` parameter followed by the field name
- Prefix with `-` for descending order
- Example: `?ordering=-amount` (descending by amount)
- Example: `?ordering=time_of_transaction` (ascending by time)

**Combined Example**:
```
/api/expenses/transactions/?category=3&min_amount=50.00&ordering=-time_of_transaction&page=2&page_size=15
```
This gets the second page of expenses in category 3, with amounts over $50, sorted by newest first, 15 items per page. 

## 4. API Endpoints

This section provides detailed documentation for all API endpoints. Each endpoint includes information about HTTP methods, URL patterns, request parameters, response formats, and authentication requirements.

### 4.1 Authentication

Authentication is handled through Firebase and requires sending a bearer token with each request as described in section 3.4.1.

#### 4.1.1 Change Password

Changes the password for the authenticated user.

```
POST /api/expenses/change-password/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "old_password": "current-password",
  "new_password": "new-password"
}
```

**Response (200 OK)**:
```json
{
  "message": "Password changed successfully"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid password format or incorrect old password
- `401 Unauthorized`: Missing or invalid authentication token

### 4.2 Expense Management

#### 4.2.1 Transactions

##### List Transactions

Retrieves a paginated list of transactions for the authenticated user.

```
GET /api/expenses/transactions/
```

**Authentication Required**: Yes

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Number of items per page (default: 20)
- `search`: Text search in description or merchant
- `category`: Filter by category ID
- `subcategory`: Filter by subcategory ID
- `transaction_type`: Filter by type ('expense', 'income')
- `start_date`: Filter transactions after this date (YYYY-MM-DD)
- `end_date`: Filter transactions before this date (YYYY-MM-DD)
- `min_amount`: Filter by minimum amount
- `max_amount`: Filter by maximum amount
- `ordering`: Field to order by (prefix with `-` for descending order)

**Response (200 OK)**:
```json
{
  "count": 243,
  "next": "http://api.example.com/api/expenses/transactions/?page=2",
  "previous": null,
  "results": [
    {
      "id": 123,
      "description": "Grocery shopping",
      "amount": "65.42",
      "merchant": "Whole Foods",
      "transaction_type": "expense",
      "time_of_transaction": "2023-04-15T14:30:00Z",
      "category": 3,
      "category_name": "Food",
      "subcategory": 12,
      "subcategory_name": "Groceries",
      "created_at": "2023-04-15T14:35:22Z",
      "updated_at": "2023-04-15T14:35:22Z"
    },
    // Additional transactions...
  ]
}
```

##### Create Transaction

Creates a new transaction.

```
POST /api/expenses/transactions/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "description": "Dinner at restaurant",
  "amount": 42.75,
  "merchant": "Olive Garden",
  "transaction_type": "expense",
  "time_of_transaction": "2023-04-15T19:30:00Z",
  "category": 3,
  "subcategory": 14
}
```

**Required Fields**:
- `description`: String
- `amount`: Decimal
- `transaction_type`: String ('expense' or 'income')
- `time_of_transaction`: DateTime (ISO format)

**Optional Fields**:
- `merchant`: String
- `category`: Integer (Category ID)
- `subcategory`: Integer (SubCategory ID)

**Response (201 Created)**:
```json
{
  "id": 124,
  "description": "Dinner at restaurant",
  "amount": "42.75",
  "merchant": "Olive Garden",
  "transaction_type": "expense",
  "time_of_transaction": "2023-04-15T19:30:00Z",
  "category": 3,
  "category_name": "Food",
  "subcategory": 14,
  "subcategory_name": "Restaurants",
  "created_at": "2023-04-15T20:15:10Z",
  "updated_at": "2023-04-15T20:15:10Z"
}
```

##### Retrieve Transaction

Retrieves a specific transaction by ID.

```
GET /api/expenses/transactions/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Transaction ID

**Response (200 OK)**:
```json
{
  "id": 124,
  "description": "Dinner at restaurant",
  "amount": "42.75",
  "merchant": "Olive Garden",
  "transaction_type": "expense",
  "time_of_transaction": "2023-04-15T19:30:00Z",
  "category": 3,
  "category_name": "Food",
  "subcategory": 14,
  "subcategory_name": "Restaurants",
  "created_at": "2023-04-15T20:15:10Z",
  "updated_at": "2023-04-15T20:15:10Z"
}
```

##### Update Transaction

Updates an existing transaction.

```
PUT /api/expenses/transactions/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Transaction ID

**Request Body**: Same as Create Transaction

**Response (200 OK)**: Updated transaction object

##### Partial Update Transaction

Updates only specified fields of an existing transaction.

```
PATCH /api/expenses/transactions/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Transaction ID

**Request Body**: Any subset of transaction fields

**Response (200 OK)**: Updated transaction object

##### Delete Transaction

Deletes a transaction.

```
DELETE /api/expenses/transactions/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Transaction ID

**Response (204 No Content)**: Empty response

#### 4.2.2 Categories

##### List Categories

Retrieves a list of expense categories.

```
GET /api/expenses/categories/
```

**Authentication Required**: Yes

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Number of items per page (default: 50)

**Response (200 OK)**:
```json
{
  "count": 10,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 1,
      "name": "Housing",
      "description": "Rent, mortgage, utilities, etc.",
      "icon": "home",
      "color": "#4A90E2"
    },
    {
      "id": 2,
      "name": "Transportation",
      "description": "Car payments, gas, public transit, etc.",
      "icon": "car",
      "color": "#50E3C2"
    },
    // Additional categories...
  ]
}
```

##### Category Lookup

A simplified endpoint for quickly retrieving all categories.

```
GET /api/expenses/category-lookup/
```

**Authentication Required**: Yes

**Response (200 OK)**:
```json
[
  {
    "id": 1,
    "name": "Housing"
  },
  {
    "id": 2,
    "name": "Transportation"
  },
  // Additional categories...
]
```

#### 4.2.3 Subcategories

##### List Subcategories

Retrieves a list of expense subcategories.

```
GET /api/expenses/subcategories/
```

**Authentication Required**: Yes

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Number of items per page (default: 50)
- `category`: Filter by parent category ID

**Response (200 OK)**:
```json
{
  "count": 45,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 1,
      "name": "Rent",
      "description": "Monthly rent payments",
      "category": 1,
      "category_name": "Housing"
    },
    {
      "id": 2,
      "name": "Mortgage",
      "description": "Mortgage payments",
      "category": 1,
      "category_name": "Housing"
    },
    // Additional subcategories...
  ]
}
```

##### Subcategory Lookup

A simplified endpoint for quickly retrieving subcategories, optionally filtered by category.

```
GET /api/expenses/subcategory-lookup/
```

**Authentication Required**: Yes

**Query Parameters**:
- `category`: Optional category ID to filter results

**Response (200 OK)**:
```json
[
  {
    "id": 1,
    "name": "Rent",
    "category_id": 1
  },
  {
    "id": 2,
    "name": "Mortgage",
    "category_id": 1
  },
  // Additional subcategories...
]
```

### 4.3 Investment Management

#### 4.3.1 Portfolios

##### List Portfolios

Retrieves a paginated list of investment portfolios for the authenticated user.

```
GET /api/investment/portfolios/
```

**Authentication Required**: Yes

**Query Parameters**:
- `page`: Page number (default: 1)
- `page_size`: Number of items per page (default: 10)
- `risk_level`: Filter by risk level
- `investment_strategy`: Filter by strategy

**Response (200 OK)**:
```json
{
  "count": 3,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 1,
      "name": "Retirement Fund",
      "description": "Long-term retirement investments",
      "risk_level": "moderate",
      "investment_strategy": "balanced_growth",
      "total_amount": "45000.00",
      "created_at": "2023-01-15T10:30:00Z",
      "updated_at": "2023-04-10T14:15:22Z"
    },
    // Additional portfolios...
  ]
}
```

##### Create Portfolio

Creates a new investment portfolio.

```
POST /api/investment/portfolios/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "name": "Education Fund",
  "description": "Saving for college tuition",
  "risk_level": "low",
  "investment_strategy": "conservative"
}
```

**Required Fields**:
- `name`: String
- `risk_level`: String (options: 'low', 'moderate', 'high')

**Optional Fields**:
- `description`: String
- `investment_strategy`: String (options: 'conservative', 'balanced_growth', 'aggressive_growth')

**Response (201 Created)**: Created portfolio object

##### Retrieve Portfolio

Retrieves a specific portfolio by ID.

```
GET /api/investment/portfolios/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Portfolio ID

**Response (200 OK)**: Portfolio object with items

##### Update Portfolio

Updates an existing portfolio.

```
PUT /api/investment/portfolios/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Portfolio ID

**Request Body**: Same as Create Portfolio

**Response (200 OK)**: Updated portfolio object

##### Delete Portfolio

Deletes a portfolio and all associated items.

```
DELETE /api/investment/portfolios/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Portfolio ID

**Response (204 No Content)**: Empty response

#### 4.3.2 Portfolio Items

##### List Portfolio Items

Retrieves items in a specific portfolio.

```
GET /api/investment/portfolio-items/?portfolio={portfolio_id}
```

**Authentication Required**: Yes

**Query Parameters**:
- `portfolio`: Portfolio ID (required)
- `page`: Page number (default: 1)
- `page_size`: Number of items per page (default: 20)

**Response (200 OK)**:
```json
{
  "count": 5,
  "next": null,
  "previous": null,
  "results": [
    {
      "id": 1,
      "portfolio": 1,
      "asset_name": "Apple Inc.",
      "asset_symbol": "AAPL",
      "asset_type": "stock",
      "quantity": "10.000",
      "buy_price": "150.00",
      "buy_date": "2023-01-20",
      "current_price": "170.00",
      "current_value": "1700.00"
    },
    // Additional portfolio items...
  ]
}
```

##### Create Portfolio Item

Adds a new item to a portfolio.

```
POST /api/investment/portfolio-items/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "portfolio": 1,
  "asset_name": "Microsoft",
  "asset_symbol": "MSFT",
  "asset_type": "stock",
  "quantity": 5,
  "buy_price": 280.00,
  "buy_date": "2023-04-15"
}
```

**Required Fields**:
- `portfolio`: Integer (Portfolio ID)
- `asset_name`: String
- `quantity`: Decimal
- `buy_price`: Decimal
- `buy_date`: Date (YYYY-MM-DD)

**Optional Fields**:
- `asset_symbol`: String
- `asset_type`: String (default: 'stock')

**Response (201 Created)**: Created portfolio item object

##### Update Portfolio Item

Updates an existing portfolio item.

```
PUT /api/investment/portfolio-items/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Portfolio Item ID

**Request Body**: Same as Create Portfolio Item

**Response (200 OK)**: Updated portfolio item object

##### Delete Portfolio Item

Removes an item from a portfolio.

```
DELETE /api/investment/portfolio-items/{id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `id`: Portfolio Item ID

**Response (204 No Content)**: Empty response

#### 4.3.3 Portfolio Analysis

##### Portfolio Summary

Retrieves a summary of all user portfolios with asset allocation.

```
GET /api/investment/summary/
```

**Authentication Required**: Yes

**Response (200 OK)**:
```json
{
  "total_invested": 65000.00,
  "current_value": 72500.00,
  "returns": 7500.00,
  "returns_percentage": 11.53,
  "asset_allocation": [
    {
      "type": "Stocks",
      "percentage": 60.5,
      "value": 43862.50
    },
    {
      "type": "Bonds",
      "percentage": 30.2,
      "value": 21895.00
    },
    {
      "type": "Cash",
      "percentage": 9.3,
      "value": 6742.50
    }
  ]
}
```

### 4.4 AI Services

#### 4.4.1 Expense Prediction

##### Standard Prediction

Predicts the category and subcategory for an expense based on description and merchant.

```
POST /api/ai/expense/predict/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "description": "Amazon.com",
  "merchant": "Amazon"
}
```

**Required Fields**:
- `description`: String

**Optional Fields**:
- `merchant`: String

**Response (200 OK)**:
```json
{
  "predicted_category": {
    "id": 6,
    "name": "Shopping"
  },
  "predicted_subcategory": {
    "id": 24,
    "name": "Online Shopping"
  },
  "confidence": 0.87,
  "alternative_categories": [
    {
      "category": {
        "id": 9,
        "name": "Entertainment"
      },
      "confidence": 0.10
    }
  ]
}
```

##### Custom Prediction

Makes a prediction with custom parameters.

```
POST /api/ai/expense/predict/custom/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "description": "Monthly train pass",
  "amount": 150.00,
  "merchant": "Transit Authority",
  "additional_context": "Commuting expense for work"
}
```

**Required Fields**:
- `description`: String

**Optional Fields**:
- `merchant`: String
- `amount`: Decimal
- `additional_context`: String

**Response (200 OK)**: Same format as Standard Prediction

#### 4.4.2 Expense Model Training

##### Trigger Model Training

Manually triggers retraining of the expense categorization model.

```
POST /api/ai/expense/train/
```

**Authentication Required**: Yes

**Response (202 Accepted)**:
```json
{
  "message": "Model training initiated",
  "job_id": "train_20230415_123456",
  "estimated_completion_time": "2023-04-15T13:30:00Z"
}
```

##### Get Model Metrics

Retrieves performance metrics for the expense categorization model.

```
GET /api/ai/expense/metrics/
```

**Authentication Required**: Yes

**Response (200 OK)**:
```json
{
  "accuracy": 0.92,
  "precision": 0.89,
  "recall": 0.87,
  "f1_score": 0.88,
  "sample_size": 1250,
  "last_trained": "2023-04-10T09:15:30Z",
  "top_categories": [
    {
      "category": "Food",
      "accuracy": 0.95
    },
    {
      "category": "Transportation",
      "accuracy": 0.93
    }
  ]
}
```

#### 4.4.3 Investment Recommendations

##### Generate Investment Recommendations

Generates investment recommendations based on user profile.

```
POST /api/ai/investment/predict/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "user_profile": {
    "age": 35,
    "risk_tolerance": "moderate",
    "investment_horizon": "long_term",
    "income_level": "middle",
    "financial_goals": ["retirement", "home_purchase"]
  },
  "recommender_type": "advanced"
}
```

**Required Fields**:
- `user_profile`: Object containing user information

**Optional Fields**:
- `recommender_type`: String (options: 'basic', 'advanced')

**Response (200 OK)**:
```json
{
  "recommended_allocations": [
    {
      "asset_class": "Stocks",
      "percentage": 60,
      "rationale": "Long-term growth potential aligned with your age and goals"
    },
    {
      "asset_class": "Bonds",
      "percentage": 30,
      "rationale": "Stability and income generation"
    },
    {
      "asset_class": "Cash",
      "percentage": 10,
      "rationale": "Emergency fund and short-term liquidity"
    }
  ],
  "specific_recommendations": [
    {
      "name": "Total Stock Market Index Fund",
      "allocation": 40,
      "type": "index_fund",
      "risk_level": "moderate"
    },
    {
      "name": "International Stock Index Fund",
      "allocation": 20,
      "type": "index_fund",
      "risk_level": "moderate_high"
    },
    {
      "name": "Total Bond Market Fund",
      "allocation": 30,
      "type": "bond_fund",
      "risk_level": "low_moderate"
    },
    {
      "name": "Money Market Fund",
      "allocation": 10,
      "type": "money_market",
      "risk_level": "low"
    }
  ],
  "confidence": 0.85
}
```

##### Analyze Portfolio

Analyzes a specific portfolio and provides insights.

```
GET /api/ai/investment/analyze/{portfolio_id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `portfolio_id`: Portfolio ID

**Response (200 OK)**:
```json
{
  "diversification_score": 75,
  "risk_assessment": "moderate",
  "expected_annual_return": 7.5,
  "volatility": 12.3,
  "recommendations": [
    {
      "type": "rebalance",
      "description": "Consider increasing bond allocation by 5% to better align with your risk profile"
    },
    {
      "type": "diversify",
      "description": "Add international exposure to reduce concentration in domestic markets"
    }
  ],
  "performance_projection": {
    "5_year": {
      "optimistic": 146500,
      "expected": 127500,
      "conservative": 112000
    },
    "10_year": {
      "optimistic": 215000,
      "expected": 182500,
      "conservative": 155000
    }
  }
}
```

##### Expense-Based Recommendations

Gets investment recommendations based on a user's expense patterns.

```
GET /api/ai/investment/expense-recommendations/{user_id}/
```

**Authentication Required**: Yes

**URL Parameters**:
- `user_id`: User Profile ID

**Response (200 OK)**:
```json
{
  "spending_summary": {
    "monthly_average": 3200.00,
    "top_categories": [
      {"category": "Housing", "percentage": 35},
      {"category": "Food", "percentage": 20},
      {"category": "Transportation", "percentage": 15}
    ]
  },
  "savings_potential": {
    "estimated_savings": 400.00,
    "source_categories": [
      {"category": "Food", "amount": 150.00, "strategy": "Reduce dining out"},
      {"category": "Shopping", "amount": 250.00, "strategy": "Delay discretionary purchases"}
    ]
  },
  "investment_recommendations": [
    {
      "type": "regular_contribution",
      "amount": 400.00,
      "frequency": "monthly",
      "destination": "Retirement Fund",
      "impact": {
        "10_year_value": 62000,
        "annual_return": 7.5
      }
    }
  ]
}
```

#### 4.4.4 Gemini AI Integration

##### Ask Gemini

Sends a natural language query to Google Gemini and receives a response.

```
POST /api/ai/gemini/predict/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "query": "What are the best ways to save for retirement?",
  "context": {
    "user_age": 35,
    "risk_tolerance": "moderate"
  }
}
```

**Required Fields**:
- `query`: String (natural language question)

**Optional Fields**:
- `context`: Object (additional context for the question)

**Response (200 OK)**:
```json
{
  "response": "Based on your age (35) and moderate risk tolerance, consider the following retirement savings strategies:\n\n1. Maximize contributions to tax-advantaged accounts like 401(k)s and IRAs\n2. Aim for a diversified portfolio with approximately 70% in stocks and 30% in bonds\n3. Consider index funds for lower fees and broad market exposure\n4. Set up automatic contributions to ensure consistent investing\n\nStarting at age 35, you should aim to save approximately 15-20% of your income for retirement.",
  "sources": [
    {
      "title": "Retirement Planning Basics",
      "url": "https://example.com/retirement-planning"
    }
  ],
  "confidence": 0.92
}
```

##### Gemini Chat

Enables an interactive chat session with Google Gemini.

```
POST /api/ai/gemini/chat/
```

**Authentication Required**: Yes

**Request Body**:
```json
{
  "messages": [
    {"role": "user", "content": "How much should I be saving each month?"},
    {"role": "assistant", "content": "That depends on your income, expenses, and financial goals. Could you share some more details about your situation?"},
    {"role": "user", "content": "I make $5,000 per month and want to save for a house down payment"}
  ],
  "context": {
    "financial_data": {
      "monthly_income": 5000,
      "monthly_expenses": 3500
    }
  }
}
```

**Required Fields**:
- `messages`: Array of message objects with role and content

**Optional Fields**:
- `context`: Object (additional context for the conversation)

**Response (200 OK)**:
```json
{
  "reply": "Based on your monthly income of $5,000 and expenses of $3,500, you have approximately $1,500 potential savings per month.\n\nFor a house down payment, I recommend:\n\n1. Aim to save at least 20% of the home's purchase price to avoid PMI\n2. For a $300,000 home, that's $60,000\n3. At $1,500/month, it would take approximately 40 months (3.3 years)\n\nConsider putting this money in a high-yield savings account or short-term CD for safety while maintaining liquidity for your upcoming purchase.",
  "conversation_id": "chat_20230415_789012",
  "suggestions": [
    "How can I reduce my monthly expenses?",
    "What are current mortgage interest rates?",
    "Should I consider a first-time homebuyer program?"
  ]
}
```

### 4.5 API Service Information

#### 4.5.1 Service Info

Gets information about available AI services and their capabilities.

```
GET /api/ai/info/
```

**Authentication Required**: Yes

**Response (200 OK)**:
```json
{
  "services": [
    {
      "name": "expense_service",
      "version": "1.2.0",
      "last_trained": "2023-04-10T09:15:30Z",
      "capabilities": ["prediction", "custom_prediction", "training", "metrics"],
      "model_type": "random_forest",
      "accuracy": 0.92
    },
    {
      "name": "investment_service",
      "version": "1.1.5",
      "capabilities": ["prediction", "portfolio_analysis", "expense_based_recommendations"],
      "model_type": "hybrid_recommender",
      "accuracy": 0.85
    },
    {
      "name": "gemini_service",
      "version": "1.0.2",
      "capabilities": ["prediction", "chat"],
      "model_type": "gemini_pro",
      "token_limit": 30000
    }
  ],
  "system_status": "healthy",
  "api_version": "1.0"
}
```

### 4.6 Advanced API Features

#### 4.6.1 API Versioning

The API supports versioning through URL path versioning. This ensures backward compatibility as the API evolves.

**Current API Versions**:
- `v1`: Current stable version (default)
- `v2`: Reserved for future use

The API version is specified in the settings with:
```python
'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
'DEFAULT_VERSION': 'v1',
'ALLOWED_VERSIONS': ['v1', 'v2'],
'VERSION_PARAM': 'version',
```

While versioning is configured, the current implementation does not require explicit version specification in URLs. All endpoints default to v1.

**Support for Future Versions**:

Some viewsets include version-specific serializers:
```python
def get_serializer_class(self):
    """
    Returns different serializers based on API version.
    """
    if self.request.version == 'v1':
        return PortfolioSerializer
    # Default to current serializer
    return PortfolioSerializer
```

This structure allows for future API changes without breaking existing clients.

#### 4.6.2 Batch Operations

The system supports batch operations for expense predictions, though this is primarily used internally rather than exposed directly via the API.

**Batch Prediction**:

The `ExpensePredictor` class includes a `predict_categories_batch` method for processing multiple transactions at once:

```python
def predict_categories_batch(self, transactions):
    """
    Predict categories for multiple transactions at once.
    
    Args:
        transactions (list): List of transaction dictionaries
                          Each dict must contain 'description' and 'merchant'
    
    Returns:
        list: List of predictions with confidence scores
    """
```

This functionality is used internally by the system but could be exposed as a public API endpoint if needed for bulk transaction processing.

#### 4.6.3 Pagination Strategies

The API implements multiple pagination strategies for different resource types:

1. **Standard Page Number Pagination**:
   - Default for most endpoints
   - Page size: 20 items
   - Query parameters: `page` and `page_size`
   - Maximum page size: 100 items

2. **Large Results Pagination**:
   - Used for endpoints that return larger datasets
   - Page size: 50 items
   - Maximum page size: 200 items

3. **Custom Limit-Offset Pagination**:
   - Alternative pagination style available for specific endpoints
   - Default limit: 20 items
   - Maximum limit: 100 items
   - Query parameters: `limit` and `offset`

4. **Resource-Specific Pagination**:
   - Transaction-specific: Optimized for transaction listings
   - Portfolio-specific: Tailored for investment portfolios
   - Recommendation-specific: Used for recommendation results (15 items per page)

**Enhanced Pagination Response Format**:
```json
{
  "count": 243,
  "next": "http://api.example.com/api/expenses/transactions/?page=2",
  "previous": null,
  "total_pages": 13,
  "current_page": 1,
  "results": [
    // items...
  ]
}
```

#### 4.6.4 Rate Limiting

The API has rate limiting infrastructure in place, though specific limits are not currently enforced. The settings include:

```python
'DEFAULT_THROTTLE_CLASSES': [],
'DEFAULT_THROTTLE_RATES': {},
```

This configuration can be easily updated to implement rate limits when needed, particularly if the API becomes publicly accessible or experiences high traffic volumes.

**Potential Future Rate Limits**:
- Anonymous users: Stricter limits
- Authenticated users: More generous limits
- Specific endpoints (like AI services): Specialized limits to prevent overuse

#### 4.6.5 Cache Configuration

The AI services implement caching for prediction results to improve performance:

```python
# Performance settings
CACHE_PREDICTIONS = True
CACHE_TIMEOUT = 3600  # 1 hour
```

This configuration:
- Caches AI prediction results for 1 hour
- Reduces computation overhead for repeated similar requests
- Improves response times for common prediction patterns

#### 4.6.6 Cross-Origin Resource Sharing (CORS)

The API implements CORS to allow access from specified frontend origins:

```python
CORS_ALLOWED_ORIGINS = [
    'http://localhost:19006',  # Expo web
    'http://localhost:8081',   # Expo dev server
    'http://localhost:8080',   # Expo dev server
    // Additional origins...
]
```

This configuration:
- Enables cross-origin requests from approved frontend applications
- Supports credentials for authenticated requests
- Can be extended with additional origins via environment variables

#### 4.6.7 Webhooks and Event Notifications

While not currently implemented, the architecture supports the addition of webhook functionality for event notifications. Future versions may include event-driven notifications for:

- Transaction categorization updates
- Portfolio value changes
- Model retraining completion
- User profile updates

#### 4.6.8 OpenAPI/Swagger Documentation

The API includes built-in interactive documentation using Swagger/OpenAPI:

- Swagger UI: `/swagger/`
- ReDoc interface: `/redoc/`
- JSON schema: `/swagger.json`
- YAML schema: `/swagger.yaml`

These endpoints provide interactive documentation that allows frontend developers to:
- Explore available endpoints
- Test API calls directly from the browser
- View request/response schemas
- Understand authentication requirements 

## 5. Data Models

This section provides detailed information about the database models, their relationships, and validation rules. Understanding these models is essential for properly integrating with the API.

### 5.1 Expenses Models

#### 5.1.1 Transaction

The `Transaction` model represents financial transactions recorded by users.

**Fields**:
- `id`: Integer, Primary Key
- `user`: ForeignKey to User, the owner of this transaction
- `description`: String, description of the transaction
- `amount`: Decimal, the monetary amount (positive or negative)
- `merchant`: String, the vendor or merchant name (optional)
- `transaction_type`: String, choices: 'expense' or 'income'
- `time_of_transaction`: DateTime, when the transaction occurred
- `category`: ForeignKey to Category (optional)
- `subcategory`: ForeignKey to SubCategory (optional)
- `created_at`: DateTime, when the record was created
- `updated_at`: DateTime, when the record was last updated

**Validation Rules**:
- `amount` must be a valid decimal number
- `transaction_type` must be one of the predefined choices
- `time_of_transaction` must be a valid date/time
- If `subcategory` is specified, it must belong to the selected `category`

**Relationships**:
- Each Transaction belongs to one User
- Each Transaction can be assigned to one Category
- Each Transaction can be assigned to one SubCategory

#### 5.1.2 Category

The `Category` model represents top-level expense categories.

**Fields**:
- `id`: Integer, Primary Key
- `name`: String, name of the category
- `description`: String, detailed description (optional)
- `icon`: String, identifier for frontend icon (optional)
- `color`: String, color code for UI representation (optional)

**Validation Rules**:
- `name` must be unique

**Relationships**:
- One Category can have many Transactions
- One Category can have many SubCategories

#### 5.1.3 SubCategory

The `SubCategory` model represents more specific expense classifications under a parent category.

**Fields**:
- `id`: Integer, Primary Key
- `name`: String, name of the subcategory
- `description`: String, detailed description (optional)
- `category`: ForeignKey to Category, the parent category

**Validation Rules**:
- `name` must be unique within a category
- `category` is required

**Relationships**:
- Each SubCategory belongs to one Category
- One SubCategory can have many Transactions

### 5.2 Investment Models

#### 5.2.1 UserProfile

The `UserProfile` model extends the Django User model with investment-specific attributes.

**Fields**:
- `id`: Integer, Primary Key
- `user`: OneToOneField to User, the associated Django user
- `risk_tolerance`: String, choices: 'low', 'moderate', 'high'
- `investment_horizon`: String, choices: 'short_term', 'medium_term', 'long_term'
- `income_level`: String, choices: 'low', 'middle', 'high'
- `financial_goals`: JSONField, list of financial goals
- `investment_experience`: String, choices: 'beginner', 'intermediate', 'experienced'
- `age_bracket`: String, age range of the user
- `created_at`: DateTime, when the profile was created
- `updated_at`: DateTime, when the profile was last updated

**Validation Rules**:
- `risk_tolerance`, `investment_horizon`, `income_level`, and `investment_experience` must be one of the predefined choices

**Relationships**:
- Each UserProfile is associated with exactly one User
- One UserProfile can have many Portfolios

#### 5.2.2 Portfolio

The `Portfolio` model represents an investment portfolio belonging to a user.

**Fields**:
- `id`: Integer, Primary Key
- `user`: ForeignKey to UserProfile, the owner of this portfolio
- `name`: String, name of the portfolio
- `description`: String, detailed description (optional)
- `risk_level`: String, choices: 'low', 'moderate', 'high'
- `investment_strategy`: String, choices: 'conservative', 'balanced_growth', 'aggressive_growth'
- `total_amount`: Decimal, total value of the portfolio
- `created_at`: DateTime, when the portfolio was created
- `updated_at`: DateTime, when the portfolio was last updated

**Validation Rules**:
- `name` must be unique for a user
- `risk_level` and `investment_strategy` must be one of the predefined choices
- `total_amount` must be a non-negative decimal

**Relationships**:
- Each Portfolio belongs to one UserProfile
- One Portfolio can have many PortfolioItems

#### 5.2.3 PortfolioItem

The `PortfolioItem` model represents individual assets within a portfolio.

**Fields**:
- `id`: Integer, Primary Key
- `portfolio`: ForeignKey to Portfolio, the containing portfolio
- `asset_name`: String, name of the asset
- `asset_symbol`: String, ticker symbol or identifier (optional)
- `asset_type`: String, choices: 'stock', 'bond', 'mutual_fund', 'etf', 'crypto', 'cash', 'other'
- `quantity`: Decimal, number of units owned
- `buy_price`: Decimal, price per unit at purchase
- `buy_date`: Date, when the asset was purchased
- `current_price`: Decimal, current price per unit (updated periodically)
- `notes`: Text, additional notes (optional)
- `created_at`: DateTime, when the record was created
- `updated_at`: DateTime, when the record was last updated

**Validation Rules**:
- `asset_name` is required
- `asset_type` must be one of the predefined choices
- `quantity` and `buy_price` must be positive decimals
- `buy_date` must be a valid date

**Relationships**:
- Each PortfolioItem belongs to one Portfolio

#### 5.2.4 Asset

The `Asset` model represents investment assets available in the system.

**Fields**:
- `id`: Integer, Primary Key
- `name`: String, name of the asset
- `symbol`: String, ticker symbol or identifier
- `asset_type`: String, type of asset
- `risk_level`: String, choices: 'low', 'low_moderate', 'moderate', 'moderate_high', 'high'
- `description`: Text, detailed description (optional)
- `metadata`: JSONField, additional asset information

**Validation Rules**:
- `name` and `symbol` combination must be unique
- `risk_level` must be one of the predefined choices

**Relationships**:
- One Asset can be included in many PortfolioItems

### 5.3 AI Service Models

The AI Service doesn't directly define database models but instead works with:

1. **Trained ML Models**: Serialized machine learning models stored as files
   - Random Forest classifiers for expense categorization
   - Hybrid recommenders for investment suggestions

2. **Feature Vectors**: Numerical representations of text data

3. **Prediction Results**: Structured data returned from ML models

4. **Service Configurations**: Parameters controlling AI behavior

### 5.4 Data Validation

Data validation occurs at multiple levels:

1. **Database Constraints**:
   - Foreign key constraints
   - Unique constraints
   - NOT NULL constraints

2. **Django Model Validators**:
   - Field-level validators
   - Model-level validators

3. **Serializer Validation**:
   - DRF serializers enforce field validation
   - Custom validation logic for complex rules

4. **Business Logic Validation**:
   - Service layer validation for complex rules
   - Cross-field validations

**Example Validation Rules**:

```python
# Transaction model validation
def clean(self):
    # Ensure subcategory belongs to the selected category
    if self.subcategory and self.category and self.subcategory.category != self.category:
        raise ValidationError('Subcategory must belong to the selected category')
```

### 5.5 Model Inheritance

The system uses several forms of model inheritance:

1. **Abstract Base Models**: Used for common fields like created_at/updated_at

2. **Proxy Models**: Used for providing additional behavior without changing the database

3. **Multi-table Inheritance**: Used for UserProfile extending the Django User model

### 5.6 Database Migrations

The database schema is managed through Django migrations. The migrations directory in each app contains the migration history for that app's models. New models or changes to existing models require generating and applying migrations before they take effect in the database. 

## 6. Frontend Integration Guidelines

This section provides practical guidance for frontend developers on integrating with the API, covering authentication, data fetching patterns, error handling strategies, and best practices.

### 6.1 Authentication Implementation

#### 6.1.1 Firebase Authentication Setup

To implement authentication in your frontend application:

1. **Install Firebase SDK**:

```bash
# Using npm
npm install firebase

# Using yarn
yarn add firebase
```

2. **Initialize Firebase** in your application:

```javascript
// firebase.js
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
  // Other Firebase configuration
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { auth };
```

3. **Implement Authentication Functions**:

```javascript
// auth-service.js
import { 
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut,
  sendPasswordResetEmail,
  updatePassword
} from 'firebase/auth';
import { auth } from './firebase';

// Sign in
export const login = async (email, password) => {
  return signInWithEmailAndPassword(auth, email, password);
};

// Register
export const register = async (email, password) => {
  return createUserWithEmailAndPassword(auth, email, password);
};

// Sign out
export const logout = async () => {
  return signOut(auth);
};

// Reset password
export const resetPassword = async (email) => {
  return sendPasswordResetEmail(auth, email);
};

// Change password
export const changePassword = async (newPassword) => {
  const user = auth.currentUser;
  if (!user) throw new Error('No authenticated user');
  return updatePassword(user, newPassword);
};
```

#### 6.1.2 Token Management

After successful authentication, you need to obtain and manage the Firebase ID token for API requests:

```javascript
// token-service.js
import { auth } from './firebase';

// Get current auth token
export const getAuthToken = async () => {
  const user = auth.currentUser;
  if (!user) return null;
  
  return user.getIdToken(true);
};

// Refresh token before it expires
export const refreshAuthToken = async () => {
  const user = auth.currentUser;
  if (!user) return null;
  
  // Force token refresh
  return user.getIdToken(true);
};
```

#### 6.1.3 Adding Authentication to Requests

Add the authentication token to all API requests:

```javascript
// api-client.js
import axios from 'axios';
import { getAuthToken } from './token-service';

const apiClient = axios.create({
  baseURL: 'https://api.example.com/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use(async (config) => {
  const token = await getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export default apiClient;
```

### 6.2 API Client Setup

#### 6.2.1 Base API Client

Create a centralized API client to handle all requests:

```javascript
// api-client.js (expanded)
import axios from 'axios';
import { getAuthToken, refreshAuthToken } from './token-service';

const apiClient = axios.create({
  baseURL: 'https://api.example.com/api',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds
});

// Request interceptor (adds auth token)
apiClient.interceptors.request.use(async (config) => {
  const token = await getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Response interceptor (handles errors)
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // If error is 401 Unauthorized and we haven't already tried to refresh
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the token
        await refreshAuthToken();
        
        // Get new token
        const token = await getAuthToken();
        
        // Update header
        originalRequest.headers.Authorization = `Bearer ${token}`;
        
        // Retry original request
        return apiClient(originalRequest);
      } catch (refreshError) {
        // If refresh fails, redirect to login
        window.location.href = '/login';
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export default apiClient;
```

#### 6.2.2 API Service Modules

Organize API calls into logical service modules:

```javascript
// expenses-service.js
import apiClient from './api-client';

export const TransactionsService = {
  getAll: async (params = {}) => {
    return apiClient.get('/expenses/transactions/', { params });
  },
  
  getById: async (id) => {
    return apiClient.get(`/expenses/transactions/${id}/`);
  },
  
  create: async (data) => {
    return apiClient.post('/expenses/transactions/', data);
  },
  
  update: async (id, data) => {
    return apiClient.put(`/expenses/transactions/${id}/`, data);
  },
  
  delete: async (id) => {
    return apiClient.delete(`/expenses/transactions/${id}/`);
  },
  
  // Additional transaction-specific methods
};

export const CategoriesService = {
  getAll: async () => {
    return apiClient.get('/expenses/categories/');
  },
  
  getSubcategories: async (categoryId) => {
    return apiClient.get('/expenses/subcategory-lookup/', {
      params: { category: categoryId }
    });
  },
  
  // Additional category-related methods
};
```

### 6.3 Data Fetching Patterns

#### 6.3.1 React Hooks (If using React)

Implement custom hooks for data fetching:

```javascript
// use-transactions.js
import { useState, useEffect } from 'react';
import { TransactionsService } from './expenses-service';

export const useTransactions = (filters = {}) => {
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({
    count: 0,
    next: null,
    previous: null,
    currentPage: 1
  });

  const fetchTransactions = async (page = 1, pageSize = 20) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = {
        ...filters,
        page,
        page_size: pageSize
      };
      
      const response = await TransactionsService.getAll(params);
      setTransactions(response.data.results);
      setPagination({
        count: response.data.count,
        next: response.data.next,
        previous: response.data.previous,
        currentPage: response.data.current_page || page
      });
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchTransactions();
  }, [JSON.stringify(filters)]);

  // Handle pagination
  const loadNextPage = () => {
    if (pagination.next) {
      fetchTransactions(pagination.currentPage + 1);
    }
  };

  const loadPreviousPage = () => {
    if (pagination.previous) {
      fetchTransactions(pagination.currentPage - 1);
    }
  };

  return { 
    transactions, 
    loading, 
    error, 
    pagination, 
    loadNextPage, 
    loadPreviousPage,
    refetch: fetchTransactions
  };
};
```

#### 6.3.2 Async Data Management (Framework Agnostic)

For any frontend framework, implement a flexible data manager:

```javascript
// data-manager.js
export class DataManager {
  constructor(fetchFunction) {
    this.fetchFunction = fetchFunction;
    this.data = null;
    this.loading = false;
    this.error = null;
    this.lastFetched = null;
  }

  async fetch(...args) {
    this.loading = true;
    this.error = null;
    
    try {
      const response = await this.fetchFunction(...args);
      this.data = response.data;
      this.lastFetched = new Date();
      return response.data;
    } catch (err) {
      this.error = err;
      throw err;
    } finally {
      this.loading = false;
    }
  }

  get isStale() {
    if (!this.lastFetched) return true;
    
    // Consider data stale after 5 minutes
    const staleTime = 5 * 60 * 1000; 
    return (new Date() - this.lastFetched) > staleTime;
  }

  clearCache() {
    this.data = null;
    this.lastFetched = null;
  }
}

// Usage example
const transactionsManager = new DataManager(TransactionsService.getAll);
```

### 6.4 Error Handling Strategies

#### 6.4.1 Global Error Handler

Implement a global error handler:

```javascript
// error-handler.js
export const handleApiError = (error) => {
  // Network errors
  if (!error.response) {
    return {
      message: 'Network error: Please check your internet connection',
      details: error.message,
      type: 'network'
    };
  }

  // HTTP errors
  const { status, data } = error.response;
  
  switch (status) {
    case 400:
      return {
        message: 'Invalid request',
        details: data.detail || data.error || 'The request was invalid',
        fields: data.detail || {},
        type: 'validation'
      };
      
    case 401:
      return {
        message: 'Authentication required',
        details: 'Please log in to continue',
        type: 'auth'
      };
      
    case 403:
      return {
        message: 'Access denied',
        details: 'You do not have permission to perform this action',
        type: 'permission'
      };
      
    case 404:
      return {
        message: 'Not found',
        details: data.detail || 'The requested resource was not found',
        type: 'not_found'
      };
      
    case 500:
    case 502:
    case 503:
      return {
        message: 'Server error',
        details: 'Something went wrong on our end. Please try again later',
        type: 'server'
      };
      
    default:
      return {
        message: 'Something went wrong',
        details: data.detail || data.error || 'An unexpected error occurred',
        type: 'unknown'
      };
  }
};
```

#### 6.4.2 Component-Level Error Handling

Examples of handling errors in components:

```javascript
// React example
const TransactionList = () => {
  const { transactions, loading, error } = useTransactions();
  
  if (loading) return <LoadingSpinner />;
  
  if (error) {
    const errorDetails = handleApiError(error);
    return (
      <ErrorDisplay 
        message={errorDetails.message} 
        details={errorDetails.details}
        onRetry={() => refetch()}
      />
    );
  }
  
  return (
    <div className="transaction-list">
      {transactions.map(transaction => (
        <TransactionItem key={transaction.id} transaction={transaction} />
      ))}
    </div>
  );
};
```

#### 6.4.3 Form Validation Errors

Handle field-level validation errors:

```javascript
// Form submission example
const handleSubmit = async (formData) => {
  try {
    await TransactionsService.create(formData);
    // Success handling
  } catch (error) {
    if (error.response?.status === 400) {
      // Extract field errors
      const fieldErrors = error.response.data.detail || {};
      
      // Update form state with errors
      setErrors(fieldErrors);
    } else {
      // Handle other errors
      const errorDetails = handleApiError(error);
      setGlobalError(errorDetails.message);
    }
  }
};
```

### 6.5 State Management

#### 6.5.1 Caching Strategies

Implement data caching for performance:

```javascript
// cache-service.js
export class CacheService {
  constructor(ttl = 5 * 60 * 1000) { // Default 5 min TTL
    this.cache = new Map();
    this.ttl = ttl;
  }
  
  set(key, data) {
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }
  
  get(key) {
    const cached = this.cache.get(key);
    
    if (!cached) return null;
    
    // Check if expired
    if (Date.now() - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return null;
    }
    
    return cached.data;
  }
  
  invalidate(key) {
    this.cache.delete(key);
  }
  
  clear() {
    this.cache.clear();
  }
}

// Usage with API
const apiCache = new CacheService();

export const getCachedCategories = async () => {
  const cacheKey = 'categories';
  const cachedData = apiCache.get(cacheKey);
  
  if (cachedData) {
    return cachedData;
  }
  
  const response = await CategoriesService.getAll();
  apiCache.set(cacheKey, response.data);
  return response.data;
};
```

#### 6.5.2 Optimistic Updates

Implement optimistic updates for better user experience:

```javascript
// Example of optimistic update
const deleteTransaction = async (id) => {
  // Save current state for potential rollback
  const previousTransactions = [...transactions];
  
  // Optimistically update UI
  setTransactions(transactions.filter(t => t.id !== id));
  
  try {
    // Perform actual delete
    await TransactionsService.delete(id);
    // Success - nothing more to do as UI is already updated
  } catch (error) {
    // On failure, rollback to previous state
    setTransactions(previousTransactions);
    // Show error
    setError(handleApiError(error));
  }
};
```

### 6.6 Performance Optimization

#### 6.6.1 Pagination and Infinite Scrolling

Implement efficient pagination or infinite scrolling:

```javascript
// Infinite scroll example with React
const TransactionListInfinite = () => {
  const [page, setPage] = useState(1);
  const [allTransactions, setAllTransactions] = useState([]);
  const [hasMore, setHasMore] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  
  const loadMore = async () => {
    if (isLoading || !hasMore) return;
    
    setIsLoading(true);
    
    try {
      const response = await TransactionsService.getAll({ page, page_size: 20 });
      const newTransactions = response.data.results;
      
      setAllTransactions(prev => [...prev, ...newTransactions]);
      setPage(page + 1);
      setHasMore(!!response.data.next);
    } catch (error) {
      console.error(handleApiError(error));
    } finally {
      setIsLoading(false);
    }
  };
  
  // Initial load
  useEffect(() => {
    loadMore();
  }, []);
  
  return (
    <div className="transaction-list">
      {allTransactions.map(transaction => (
        <TransactionItem key={transaction.id} transaction={transaction} />
      ))}
      
      {isLoading && <LoadingIndicator />}
      
      {hasMore && !isLoading && (
        <button onClick={loadMore}>Load More</button>
      )}
    </div>
  );
};
```

#### 6.6.2 Request Batching

Batch multiple requests when possible:

```javascript
// Batch fetch example
const fetchDashboardData = async () => {
  setLoading(true);
  
  try {
    // Fetch multiple resources in parallel
    const [
      transactionsResponse, 
      portfolioResponse, 
      summaryResponse
    ] = await Promise.all([
      TransactionsService.getAll({ page_size: 5 }),
      PortfolioService.getAll(),
      PortfolioService.getSummary()
    ]);
    
    // Process all responses
    setDashboardData({
      recentTransactions: transactionsResponse.data.results,
      portfolios: portfolioResponse.data.results,
      summary: summaryResponse.data
    });
  } catch (error) {
    setError(handleApiError(error));
  } finally {
    setLoading(false);
  }
};
```

### 6.7 Testing API Integration

#### 6.7.1 Mock API Responses

Set up testing with mock API responses:

```javascript
// api-mocks.js
export const mockTransactions = [
  {
    id: 1,
    description: "Grocery shopping",
    amount: "65.42",
    merchant: "Whole Foods",
    transaction_type: "expense",
    time_of_transaction: "2023-04-15T14:30:00Z",
    category: 3,
    category_name: "Food",
    subcategory: 12,
    subcategory_name: "Groceries"
  },
  // More mock transactions...
];

export const mockApiResponses = {
  '/expenses/transactions/': {
    get: {
      status: 200,
      data: {
        count: mockTransactions.length,
        next: null,
        previous: null,
        results: mockTransactions
      }
    },
    post: {
      status: 201,
      data: mockTransactions[0]
    }
  },
  // More endpoint mocks...
};
```

#### 6.7.2 Testing with Mock API

Example of testing with mocked API:

```javascript
// transaction-list.test.js
import { render, screen, waitFor } from '@testing-library/react';
import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import TransactionList from './TransactionList';
import { mockTransactions, mockApiResponses } from './api-mocks';

// Create axios mock
const mockAxios = new MockAdapter(axios);

describe('TransactionList', () => {
  beforeEach(() => {
    // Set up mock response for transactions endpoint
    mockAxios.onGet('/api/expenses/transactions/').reply(200, mockApiResponses['/expenses/transactions/'].get.data);
  });
  
  afterEach(() => {
    mockAxios.reset();
  });
  
  test('renders transactions when API returns data', async () => {
    render(<TransactionList />);
    
    // Should show loading initially
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    
    // Wait for transactions to load
    await waitFor(() => {
      expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument();
    });
    
    // Should display transactions
    expect(screen.getByText('Grocery shopping')).toBeInTheDocument();
    expect(screen.getByText('$65.42')).toBeInTheDocument();
  });
  
  test('handles API error', async () => {
    // Override the mock to return an error
    mockAxios.onGet('/api/expenses/transactions/').reply(500);
    
    render(<TransactionList />);
    
    // Wait for error to display
    await waitFor(() => {
      expect(screen.getByText('Server error')).toBeInTheDocument();
    });
  });
});
```

### 6.8 Mobile Integration Considerations

If your frontend includes a mobile app (React Native, Flutter, etc.), consider these additional points:

#### 6.8.1 Offline Support

Implement offline capabilities:

```javascript
// offline-queue.js
export class OfflineQueue {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
  }
  
  addToQueue(request) {
    this.queue.push({
      ...request,
      timestamp: Date.now()
    });
    
    // Persist queue to storage
    this.saveQueue();
  }
  
  async processQueue() {
    if (this.isProcessing || this.queue.length === 0) return;
    
    this.isProcessing = true;
    
    while (this.queue.length > 0) {
      const request = this.queue[0];
      
      try {
        await this.executeRequest(request);
        // Remove successful request from queue
        this.queue.shift();
        this.saveQueue();
      } catch (error) {
        // If it's a network error, stop processing
        if (!error.response) {
          break;
        }
        
        // For other errors, remove from queue and continue
        this.queue.shift();
        this.saveQueue();
      }
    }
    
    this.isProcessing = false;
  }
  
  async executeRequest(request) {
    const { method, url, data } = request;
    return apiClient({
      method,
      url,
      data
    });
  }
  
  saveQueue() {
    // Save queue to local storage or AsyncStorage
    localStorage.setItem('offlineQueue', JSON.stringify(this.queue));
  }
  
  loadQueue() {
    // Load queue from storage
    const saved = localStorage.getItem('offlineQueue');
    if (saved) {
      this.queue = JSON.parse(saved);
    }
  }
}

// Usage
const offlineQueue = new OfflineQueue();

// When online status changes
window.addEventListener('online', () => {
  offlineQueue.processQueue();
});
```

#### 6.8.2 Optimizing for Mobile Networks

Reduce data usage for mobile networks:

```javascript
// Configure API client with network-aware settings
const configureApiForMobile = (isMeteredConnection) => {
  if (isMeteredConnection) {
    // Reduce image quality, limit page sizes
    apiClient.defaults.params = {
      ...apiClient.defaults.params,
      page_size: 10,
      image_quality: 'low'
    };
  } else {
    // Use higher quality on WiFi
    apiClient.defaults.params = {
      ...apiClient.defaults.params,
      page_size: 20,
      image_quality: 'high'
    };
  }
};
```

### 6.9 Security Considerations

#### 6.9.1 Sensitive Data Handling

Properly handle sensitive financial data:

```javascript
// Mask sensitive data in logs
const maskSensitiveData = (data) => {
  if (!data) return data;
  
  const maskedData = { ...data };
  
  // Mask account numbers, etc.
  if (maskedData.accountNumber) {
    maskedData.accountNumber = maskedData.accountNumber.replace(/\d(?=\d{4})/g, '*');
  }
  
  return maskedData;
};

// Example usage with logging
const logApiCall = (endpoint, data) => {
  console.log(`API Call to ${endpoint}`, maskSensitiveData(data));
};
```

#### 6.9.2 Secure Storage

Store tokens and sensitive data securely:

```javascript
// Secure storage service
export const SecureStorage = {
  // Save token securely
  setToken: (token) => {
    // Use secure storage mechanism
    // For web: Use HttpOnly cookies or encrypted localStorage
    // For mobile: Use SecureStore or Keychain
    localStorage.setItem('auth_token', encrypt(token));
  },
  
  // Get token
  getToken: () => {
    const encryptedToken = localStorage.getItem('auth_token');
    if (!encryptedToken) return null;
    return decrypt(encryptedToken);
  },
  
  // Clear token
  clearToken: () => {
    localStorage.removeItem('auth_token');
  }
};

// Simple encryption (example only - use a proper encryption library)
function encrypt(text) {
  // Implement proper encryption
  return btoa(text); // This is NOT secure, just an example
}

function decrypt(encryptedText) {
  // Implement proper decryption
  return atob(encryptedText); // This is NOT secure, just an example
}
```

## 7. Example Requests and Responses

This section provides complete examples of common API operations based on the actual implementation in the codebase. All examples reference real endpoints found in the `expenses`, `investment`, and `ai_service` apps. These examples can be used as a reference when implementing API integration.

> **Note**: These examples reflect the actual API structure as implemented in the Django views and URLs of this project, following the endpoints defined in `expenses/urls.py`, `investment/urls.py`, `ai_service/urls.py`, and the main URL configuration in `ExpenseCategorizationAPI/urls.py`.

## 8. Deployment Guidelines

This section provides guidelines for deploying the API in various environments.

### 8.1 Prerequisites

Before deploying, ensure you have:

- Python 3.9+ installed
- PostgreSQL database
- Firebase project and service account credentials
- Google Gemini API key (for AI features)
- Docker (optional, for containerized deployment)

### 8.2 Environment Configuration

The application uses environment variables for configuration. Create a `.env` file based on the following template:

```
# Django settings
DEBUG=False
SECRET_KEY=your_secure_secret_key
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com

# Database settings
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_secure_password
DB_HOST=your_database_host
DB_PORT=5432

# Firebase settings
FIREBASE_SERVICE_ACCOUNT_PATH=./serviceAccountKey.json
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
FIREBASE_APP_ID=your_firebase_app_id

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-pro

# CORS settings
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 8.3 Local Development Deployment

For local development:

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file
5. Run migrations:
   ```bash
   python manage.py migrate
   ```
6. Start the development server:
   ```bash
   python manage.py runserver
   ```

### 8.4 Docker Deployment

The project includes a Dockerfile for containerized deployment:

1. Build the Docker image:
   ```bash
   docker build -t expense-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env expense-api
   ```

For Docker Compose deployment, create a `docker-compose.yml` file:

```yaml
version: '3'

services:
  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data/
    env_file:
      - ./.env.db
    ports:
      - "5432:5432"

  web:
    build: .
    command: gunicorn ExpenseCategorizationAPI.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
    ports:
      - "8000:8000"
    env_file:
      - ./.env
    depends_on:
      - db

volumes:
  postgres_data:
```

### 8.5 Production Deployment

For production environments:

1. Use a production-grade web server:
   ```bash
   pip install gunicorn
   gunicorn ExpenseCategorizationAPI.wsgi:application --bind 0.0.0.0:8000
   ```

2. Set up a reverse proxy (Nginx example):
   ```
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. Enable HTTPS using Let's Encrypt or other SSL certificate providers

4. Configure database connection pooling for better performance

5. Set up monitoring and logging (e.g., Prometheus, Grafana, ELK stack)

### 8.6 Scaling Considerations

As your application grows:

1. **Horizontal Scaling**: Deploy multiple API instances behind a load balancer
2. **Database Optimization**: 
   - Use connection pooling
   - Consider read replicas for query-heavy workloads
   - Implement caching for frequently accessed data

3. **Caching Strategy**:
   - Use Redis or Memcached for caching
   - Cache API responses for frequently accessed endpoints
   - Implement proper cache invalidation

4. **Asynchronous Processing**:
   - Use Celery for background tasks and periodic operations
   - Implement message queues for better handling of resource-intensive operations

### 8.7 Security Considerations

For secure deployment:

1. Keep all secrets in environment variables, never in code
2. Regularly update dependencies to patch security vulnerabilities
3. Implement rate limiting to prevent abuse
4. Use network isolation in production environments
5. Regularly audit and rotate access credentials
6. Implement proper CORS configuration
7. Configure Django's security middleware settings:
   - Always use HTTPS
   - Set secure cookies
   - Configure Content Security Policy
   - Implement proper XSS and CSRF protection

### 8.8 Monitoring and Maintenance

For ongoing maintenance:

1. Set up health check endpoints
2. Implement comprehensive logging
3. Configure automated backups for the database
4. Set up alerts for system anomalies
5. Create a disaster recovery plan
6. Implement a CI/CD pipeline for automated testing and deployment