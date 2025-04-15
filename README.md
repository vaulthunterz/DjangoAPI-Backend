# Expense Categorization and Investment RecommendationAPI

This is the backend API for the Expense Categorization and Investment Recommendation application, built with Django REST Framework.

## Features

- User authentication with JWT
- Expense tracking and categorization
- Machine learning for expense prediction
- Investment recommendation engine
- API endpoints for financial data visualization
- SMS transaction detection

## Setup Guide

Follow these steps to set up and run the backend server:

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv)

### Setup Commands(WIN)

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# For Windows:
& ".\.venv\Scripts\Activate.ps1"

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Create/Copy  .env file for environment variables
To be provided and copied at the root of the project.

# 5. Run initial database migrations
python manage.py migrate

# 5.1 Run  database migrations if you change the DB models
python manage.py makemigrations
python manage.py migrate

# 6. (Optional) Create a superuser for admin access
python manage.py createsuperuser

# 7. Run the development server( Where to access the server)
python manage.py runserver
OR
python manage.py runserver 'URL' eg. (python manage.py runserver 8000)

# Optional: If you want to run tests
# python manage.py test

# Optional: If you want to use ngrok for external access testing
# ngrok http 8000
```

### Additional Setup Notes

1. **Database Configuration**:
   - The default django setup uses SQLite. If you want to use PostgreSQL or MySQL, update the DATABASE_URL in the .env file.
   - For PostgreSQL: `DATABASE_URL=postgres://user:password@localhost:5432/dbname`
   - For MySQL: `DATABASE_URL=mysql://user:password@localhost:3306/dbname`

2. **Environment Variables**:
   - Make sure to copy the .env provided

3. **CORS Settings**:
   - The default CORS settings allow connections from common local development ports.
   - Add additional origins as needed for your frontend.

4. **API Documentation**:
   - After starting the server, you can access the API documentation at:
     - Swagger UI: `http://localhost:8080/swagger/`


5. **Troubleshooting**:
   - If you encounter any package conflicts, try creating a fresh virtual environment.
   - For database connection issues, verify your database credentials and ensure the database server is running.
   - For permission issues with file operations, check that your user has appropriate permissions.

6. **(Optional)Development Tools**:
   - To enable Django Debug Toolbar (if installed): Add `INTERNAL_IPS=127.0.0.1` to your .env file.
   - For better debugging: `pip install ipython django-extensions` and add 'django_extensions' to INSTALLED_APPS.


## Architecture

The backend follows a modular architecture with the following components:

- **Authentication**: JWT-based authentication system
- **Expenses App**: Core functionality for expense tracking
- **Investment App**: Investment recommendation engine
- **AI Service**: Serves the modules with ML categorization and recommendations
- **API Layer**: REST API endpoints using Django REST Framework

## License

[TO DO]

## Contributors

[TO DO]
