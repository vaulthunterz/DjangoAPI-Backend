"""
Views for displaying model metrics and visualizations
"""

import os
import sys
import json
import logging
import numpy as np
import time
import threading
import joblib
from sklearn.pipeline import Pipeline
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib import messages

logger = logging.getLogger(__name__)

def get_model_metrics():
    """Load model metrics from the saved files"""
    try:
        # Define paths to model files
        models_dir = os.path.join(settings.BASE_DIR, 'expenses', 'ml', 'models')

        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Check for model files
        category_model_path = os.path.join(models_dir, 'category_model.joblib')
        metadata_path = os.path.join(models_dir, 'metadata.joblib')
        performance_path = os.path.join(models_dir, 'performance.joblib')

        # Fallback to pickle files if joblib files don't exist
        category_model_pkl_path = os.path.join(models_dir, 'category_model.pkl')
        metadata_pkl_path = os.path.join(models_dir, 'metadata.pkl')
        performance_pkl_path = os.path.join(models_dir, 'performance.pkl')

        # Check if we have the necessary model files
        has_joblib_models = os.path.exists(category_model_path) and os.path.exists(metadata_path)
        has_pickle_models = os.path.exists(category_model_pkl_path) and os.path.exists(metadata_pkl_path)

        # Determine which performance file to use
        use_joblib_perf = os.path.exists(performance_path)
        use_pickle_perf = os.path.exists(performance_pkl_path)

        # Log the paths for debugging
        logger.info(f"Looking for model files: category_model={os.path.exists(category_model_path)}, metadata={os.path.exists(metadata_path)}, performance={os.path.exists(performance_path)}")

        # Check if we have the necessary model files
        if not (has_joblib_models or has_pickle_models):
            logger.warning("No model files found")

            # Check if there are any files in the models directory
            if os.path.exists(models_dir):
                files = os.listdir(models_dir)
                logger.info(f"Files in models directory: {files}")
            else:
                logger.warning(f"Models directory does not exist: {models_dir}")

            return {
                'error': 'Model files not found. Please train the model first.',
                'status': 'error'
            }

        # If we have model files but no performance file, we'll create a basic performance object
        if not (use_joblib_perf or use_pickle_perf):
            logger.warning("No performance file found, but model files exist. Creating basic performance metrics.")
            performance = {
                'category_accuracy': 0.0,
                'category_precision': 0.0,
                'category_recall': 0.0,
                'category_f1': 0.0,
                'individual_model_metrics': [],
                'subcategory_accuracies': {},
                'training_time': 0.0
            }

        # Load performance metrics if available
        if not (use_joblib_perf or use_pickle_perf):
            # We already created a basic performance object above
            logger.info("Using basic performance metrics")
        else:
            try:
                if use_joblib_perf:
                    logger.info(f"Loading performance data from {performance_path}")
                    # Log file size for debugging
                    if os.path.exists(performance_path):
                        performance_size = os.path.getsize(performance_path)
                        logger.info(f"Performance file size: {performance_size} bytes")
                    performance = joblib.load(performance_path)
                else:
                    logger.info(f"Loading performance data from {performance_pkl_path}")
                    # Log file size for debugging
                    if os.path.exists(performance_pkl_path):
                        performance_size = os.path.getsize(performance_pkl_path)
                        logger.info(f"Performance file size: {performance_size} bytes")
                    import pickle
                    with open(performance_pkl_path, 'rb') as f:
                        performance = pickle.load(f)
                logger.info(f"Loaded performance data: {type(performance)}")
            except Exception as e:
                logger.error(f"Error loading performance data: {str(e)}")
                # Create a basic performance object as fallback
                logger.warning("Creating basic performance metrics as fallback")
                performance = {
                    'category_accuracy': 0.0,
                    'category_precision': 0.0,
                    'category_recall': 0.0,
                    'category_f1': 0.0,
                    'individual_model_metrics': [],
                    'subcategory_accuracies': {},
                    'training_time': 0.0
                }

        # Load metadata
        try:
            if has_joblib_models:
                logger.info(f"Loading metadata from {metadata_path}")
                # Log file size for debugging
                if os.path.exists(metadata_path):
                    metadata_size = os.path.getsize(metadata_path)
                    logger.info(f"Metadata file size: {metadata_size} bytes")
                metadata = joblib.load(metadata_path)
            else:
                logger.info(f"Loading metadata from {metadata_pkl_path}")
                # Log file size for debugging
                if os.path.exists(metadata_pkl_path):
                    metadata_size = os.path.getsize(metadata_pkl_path)
                    logger.info(f"Metadata file size: {metadata_size} bytes")
                import pickle
                with open(metadata_pkl_path, 'rb') as f:
                    metadata = pickle.load(f)
            logger.info(f"Loaded metadata: {type(metadata)}")
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            return {
                'error': f"Error loading metadata: {str(e)}",
                'status': 'error'
            }

        # Log the contents of the performance and metadata for debugging
        logger.info(f"Performance keys: {performance.keys() if isinstance(performance, dict) else 'Not a dictionary'}")
        logger.info(f"Metadata keys: {metadata.keys() if isinstance(metadata, dict) else 'Not a dictionary'}")

        # Convert numpy values to Python types for JSON serialization
        try:
            # Convert accuracy values to percentages (0-100 instead of 0-1)
            category_accuracy = float(performance.get('category_accuracy', 0)) * 100
            logger.info(f"Category accuracy: {category_accuracy}%")

            # Get additional metrics if available
            category_precision = float(performance.get('category_precision', 0)) * 100
            category_recall = float(performance.get('category_recall', 0)) * 100
            category_f1 = float(performance.get('category_f1', 0)) * 100

            # Get individual model metrics if available
            individual_model_metrics = performance.get('individual_model_metrics', [])
            # Convert to percentages
            for model_metric in individual_model_metrics:
                for key in ['accuracy', 'precision', 'recall', 'f1_score']:
                    if key in model_metric:
                        model_metric[key] = float(model_metric[key]) * 100

            # Process subcategory accuracies and convert to percentages
            subcategory_accuracies = {}
            for category, accuracy in performance.get('subcategory_accuracies', {}).items():
                subcategory_accuracies[category] = float(accuracy) * 100

            # Sort subcategory accuracies by value (descending)
            sorted_subcategory_accuracies = dict(sorted(
                subcategory_accuracies.items(),
                key=lambda x: x[1],
                reverse=True
            ))

            # Calculate average subcategory accuracy (already in percentage)
            avg_subcategory_accuracy = np.mean(list(subcategory_accuracies.values())) if subcategory_accuracies else 0
            logger.info(f"Average subcategory accuracy: {avg_subcategory_accuracy}%")

            # Get training time
            training_time = float(performance.get('training_time', 0))
            training_time_minutes = training_time / 60
            logger.info(f"Training time: {training_time_minutes} minutes")

            # Get top and bottom performing categories
            top_categories = list(sorted_subcategory_accuracies.items())[:5]
            bottom_categories = list(sorted_subcategory_accuracies.items())[-5:]

            # Check if we have a model type flag
            model_type = 'unknown'
            model_type_path = os.path.join(models_dir, 'model_type.txt')
            if os.path.exists(model_type_path):
                try:
                    with open(model_type_path, 'r') as f:
                        model_type = f.read().strip()
                except Exception as e:
                    logger.error(f"Error reading model type: {str(e)}")

            # Determine if this is an ensemble model
            is_ensemble = model_type == 'ensemble'
            logger.info(f"Model type: {model_type}, Is ensemble: {is_ensemble}")

            # Extract feature importance if available
            feature_importance = {}
            try:
                if has_joblib_models:
                    # Load the category model to extract feature importance
                    category_model = joblib.load(category_model_path)

                    # Try to get feature importance from the model
                    if hasattr(category_model, '_get_feature_importance'):
                        # Direct method if available
                        feature_importance = category_model._get_feature_importance()
                        logger.info(f"Extracted feature importance using model's _get_feature_importance method")
                    elif hasattr(category_model, 'feature_importances_'):
                        # Direct feature importances
                        feature_importance = {f"feature_{i}": float(imp) for i, imp in enumerate(category_model.feature_importances_)}
                        logger.info(f"Extracted feature importance directly from model.feature_importances_")
                    elif isinstance(category_model, Pipeline) and hasattr(category_model.steps[-1][1], 'feature_importances_'):
                        # Pipeline with feature importances in the last step
                        feature_importance = {f"feature_{i}": float(imp) for i, imp in enumerate(category_model.steps[-1][1].feature_importances_)}
                        logger.info(f"Extracted feature importance from pipeline's last step")

                    # If we have feature importance but no descriptive names, try to load the feature extractor
                    if feature_importance and all(k.startswith('feature_') for k in list(feature_importance.keys())[:5]):
                        try:
                            # Load feature extractor
                            feature_extractor_path = os.path.join(models_dir, 'feature_extractor.joblib')
                            if os.path.exists(feature_extractor_path):
                                feature_extractor = joblib.load(feature_extractor_path)

                                # Get feature descriptions if available
                                if hasattr(feature_extractor, 'get_feature_descriptions'):
                                    descriptions = feature_extractor.get_feature_descriptions()

                                    # Replace feature names with actual words
                                    descriptive_importance = {}
                                    for feature, importance in feature_importance.items():
                                        if feature in descriptions:
                                            # Use the actual word as the feature name
                                            word = descriptions[feature]
                                            descriptive_importance[word] = importance
                                        else:
                                            # If no description available, extract the most meaningful part
                                            # of the feature name (usually after the last underscore)
                                            if '_' in feature:
                                                word = feature.split('_')[-1]
                                                descriptive_importance[word] = importance
                                            else:
                                                descriptive_importance[feature] = importance

                                    feature_importance = descriptive_importance
                                    logger.info(f"Replaced feature names with descriptive names")
                        except Exception as e:
                            logger.error(f"Error loading feature extractor for descriptions: {str(e)}")

                    # Sort feature importance by value (descending) and limit to top 20
                    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20])
                    logger.info(f"Extracted {len(feature_importance)} feature importances")
                else:
                    logger.warning("Cannot extract feature importance: model files not in joblib format")
            except Exception as e:
                logger.error(f"Error extracting feature importance: {str(e)}")

            # Check for historical metrics
            historical_metrics = []
            history_dir = os.path.join(models_dir, 'history')
            if not os.path.exists(history_dir):
                os.makedirs(history_dir, exist_ok=True)

            # Save current metrics to history
            current_timestamp = int(time.time())
            current_metrics = {
                'timestamp': current_timestamp,
                'date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_timestamp)),
                'model_type': model_type,
                'category_accuracy': float(category_accuracy) / 100,  # Convert back to 0-1 scale
                'category_precision': float(category_precision) / 100,
                'category_recall': float(category_recall) / 100,
                'category_f1': float(category_f1) / 100,
                'avg_subcategory_accuracy': float(avg_subcategory_accuracy) / 100,
                'training_time': float(training_time)
            }

            # Save current metrics to history file
            try:
                history_file = os.path.join(history_dir, f'metrics_{current_timestamp}.joblib')
                joblib.dump(current_metrics, history_file)
                logger.info(f"Saved current metrics to history: {history_file}")
            except Exception as e:
                logger.error(f"Error saving metrics to history: {str(e)}")

            # Load historical metrics (last 5)
            try:
                history_files = sorted([f for f in os.listdir(history_dir) if f.startswith('metrics_')], reverse=True)
                for i, file in enumerate(history_files[:5]):  # Get last 5 metrics
                    if i == 0:  # Skip current metrics
                        continue
                    try:
                        metrics_file = os.path.join(history_dir, file)
                        hist_metrics = joblib.load(metrics_file)
                        historical_metrics.append(hist_metrics)
                    except Exception as e:
                        logger.error(f"Error loading historical metrics {file}: {str(e)}")
            except Exception as e:
                logger.error(f"Error listing historical metrics: {str(e)}")

            # Convert historical metrics for display (back to percentage)
            for metrics in historical_metrics:
                for key in ['category_accuracy', 'category_precision', 'category_recall',
                           'category_f1', 'avg_subcategory_accuracy']:
                    if key in metrics:
                        metrics[key] = metrics[key] * 100

            # Get category distribution
            categories = list(metadata.get('categories', []))
            logger.info(f"Number of categories: {len(categories)}")

            # Count subcategories per category
            subcategory_counts = {}
            for category, subcategories in metadata.get('subcategories', {}).items():
                subcategory_counts[category] = len(subcategories)

            # Sort subcategory counts by value (descending)
            sorted_subcategory_counts = dict(sorted(
                subcategory_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ))

            return {
                'category_accuracy': category_accuracy,
                'category_precision': category_precision,
                'category_recall': category_recall,
                'category_f1': category_f1,
                'individual_model_metrics': individual_model_metrics,
                'subcategory_accuracies': sorted_subcategory_accuracies,
                'avg_subcategory_accuracy': avg_subcategory_accuracy,
                'top_categories': top_categories,
                'bottom_categories': bottom_categories,
                'categories': categories,
                'subcategory_counts': sorted_subcategory_counts,
                'training_time': training_time,
                'training_time_minutes': training_time_minutes,
                'model_type': model_type,
                'is_ensemble': is_ensemble,
                'feature_importance': feature_importance,
                'historical_metrics': historical_metrics,
                'current_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error processing model metrics: {str(e)}")
            return {
                'error': f"Error processing model metrics: {str(e)}",
                'status': 'error'
            }
    except Exception as e:
        logger.error(f"Error loading model metrics: {str(e)}")
        return {
            'error': f"Error loading model metrics: {str(e)}",
            'status': 'error'
        }

# Global variable to track training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None
}

def train_model_task(use_enhanced=True, sample_size=None):
    """Background task to train the model"""
    global training_status

    try:
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Initializing training...'
        training_status['start_time'] = time.time()

        # Change to the ML directory to ensure imports work correctly
        original_dir = os.getcwd()
        ml_dir = os.path.join(settings.BASE_DIR, 'expenses', 'ml')
        os.chdir(ml_dir)

        # Add the ML directory to the Python path
        if ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)

        # Get the data path
        training_data_dir = os.path.join(ml_dir, 'training_data')
        data_path = os.path.join(training_data_dir, 'transactions.csv')

        # Create the training_data directory if it doesn't exist
        os.makedirs(training_data_dir, exist_ok=True)

        # Check if the data file exists, create a sample if not
        if not os.path.exists(data_path):
            training_status['message'] = 'Creating sample training data...'
            # Create a simple CSV with headers
            with open(data_path, 'w') as f:
                f.write('description,merchant,amount,category,subcategory,is_expense\n')
                f.write('Sample transaction,Sample merchant,100,Food,Groceries,1\n')

        # Update progress
        training_status['progress'] = 10

        # Run the appropriate training script using subprocess
        import subprocess

        # Use the virtual environment Python interpreter
        venv_dir = r"C:\Users\Bedan\PycharmProjects\1 Python\ExpeCatAPI\.venv"
        venv_python = os.path.join(venv_dir, 'Scripts', 'python.exe')
        venv_activate = os.path.join(venv_dir, 'Scripts', 'Activate.ps1')

        # Check if the virtual environment Python exists
        if os.path.exists(venv_python):
            logger.info(f"Using virtual environment Python: {venv_python}")
            python_executable = venv_python

            # Try to activate the virtual environment first
            try:
                # Create a batch file to activate the virtual environment and run the training script
                batch_file = os.path.join(ml_dir, 'run_training.bat')
                with open(batch_file, 'w') as f:
                    f.write(f'@echo off\n')
                    f.write(f'cd "{ml_dir}"\n')
                    f.write(f'call "{os.path.join(venv_dir, "Scripts", "activate.bat")}"\n')
                    f.write(f'echo Virtual environment activated\n')
                    f.write(f'echo Current directory: %CD%\n')
                    f.write(f'echo Python executable: %VIRTUAL_ENV%\\Scripts\\python.exe\n')
                    f.write(f'echo PYTHONPATH: %PYTHONPATH%\n')
                    f.write(f'echo Running pip list...\n')
                    f.write(f'pip list\n')
                    f.write(f'echo Running training script...\n')

                logger.info(f"Created batch file: {batch_file}")
            except Exception as e:
                logger.error(f"Error creating batch file: {str(e)}")
        else:
            logger.warning(f"Virtual environment Python not found at {venv_python}, using system Python")
            python_executable = sys.executable

        # Flag to track if we need to use the fallback approach
        use_fallback = False

        # Train the model directly using the ExpenseCategorizer class
        training_status['message'] = 'Training model using ExpenseCategorizer...'

        try:
            # Import the ExpenseCategorizer class
            from expenses.ml.core.model import ExpenseCategorizer

            # Create the models directory if it doesn't exist
            models_dir = os.path.join(ml_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)

            # Initialize the model with appropriate settings based on user selection
            if use_enhanced:
                # Enhanced model uses the ensemble approach
                training_status['message'] = 'Initializing enhanced model (ensemble)...'
                categorizer = ExpenseCategorizer(models_dir=models_dir, use_ensemble=True)

                # Save a flag to indicate this is an ensemble model
                with open(os.path.join(models_dir, 'model_type.txt'), 'w') as f:
                    f.write('ensemble')
            else:
                # Original model uses only RandomForest
                training_status['message'] = 'Initializing original model (RandomForest only)...'
                categorizer = ExpenseCategorizer(models_dir=models_dir, use_ensemble=False)

                # Save a flag to indicate this is a RandomForest model
                with open(os.path.join(models_dir, 'model_type.txt'), 'w') as f:
                    f.write('randomforest')

            # Determine the data path
            if sample_size and sample_size > 0:
                training_status['message'] = f'Training with sample size {sample_size}...'
                # We'll use the sample data we created earlier
                data_path = os.path.join(training_data_dir, 'transactions.csv')
            else:
                if use_enhanced:
                    training_status['message'] = 'Training enhanced model with full dataset...'
                else:
                    training_status['message'] = 'Training original model with full dataset...'
                data_path = os.path.join(training_data_dir, 'transactions.csv')

            # Train the model
            training_status['progress'] = 30
            results = categorizer.train(data_path=data_path)

            # Update progress
            training_status['progress'] = 90
            training_status['message'] = 'Model training completed successfully!'

            # Log the results
            logger.info(f"Model training results: {results}")

            # Complete the training
            training_status['progress'] = 100
            training_status['end_time'] = time.time()

            # Calculate training time
            training_time = training_status['end_time'] - training_status['start_time']
            logger.info(f"Model training completed in {training_time:.2f} seconds")

            # Change back to the original directory
            os.chdir(original_dir)

            # We're done, no need for fallback
            return

        except Exception as e:
            logger.error(f"Error training model directly: {str(e)}")
            training_status['message'] = f'Error training model: {str(e)}'

            # We need to use the fallback approach
            use_fallback = True

        # Only use the fallback approach if needed
        if use_fallback:
            # Fallback to the old approach using subprocess if direct training fails
            training_status['message'] = 'Falling back to subprocess training...'

            if use_enhanced:
                if sample_size:
                    script_name = 'train_with_sample.py'
                    script_args = ['--size', str(sample_size)]
                else:
                    script_name = 'train_enhanced_model.py'
                    script_args = []
            else:
                script_name = 'train_model.py'
                script_args = []

            # Use a direct approach with the virtual environment Python
            cmd = [python_executable, os.path.join(ml_dir, script_name)] + script_args
            logger.info(f"Using direct command to run training: {' '.join(cmd)}")

            # Check if the training data directory exists and has files
            if not os.path.exists(training_data_dir) or not os.listdir(training_data_dir):
                # Create a more substantial sample dataset
                training_status['message'] = 'Creating sample training data...'
                logger.info("Creating sample training data...")

                # Create a more substantial sample dataset with multiple categories
                sample_data = [
                    'description,merchant,amount,category,subcategory,is_expense',
                    'Grocery shopping,Walmart,120.50,Food,Groceries,1',
                    'Monthly rent payment,Landlord,1200.00,Housing,Rent,1',
                    'Salary deposit,Employer,3000.00,Income,Salary,0',
                    'Gas station,Shell,45.75,Transportation,Fuel,1',
                    'Restaurant dinner,Olive Garden,65.30,Food,Dining Out,1',
                    'Phone bill,Verizon,85.99,Utilities,Phone,1',
                    'Internet service,Comcast,75.00,Utilities,Internet,1',
                    'Movie tickets,AMC Theaters,25.00,Entertainment,Movies,1',
                    'Coffee shop,Starbucks,4.50,Food,Coffee,1',
                    'Gym membership,Planet Fitness,20.00,Health,Fitness,1',
                    'Doctor visit,Medical Center,150.00,Health,Medical,1',
                    'Online shopping,Amazon,95.60,Shopping,Online,1',
                    'Clothing store,Gap,120.00,Shopping,Clothing,1',
                    'Car insurance,Geico,110.00,Insurance,Auto,1',
                    'Health insurance,Blue Cross,200.00,Insurance,Health,1',
                    'Freelance payment,Client,500.00,Income,Freelance,0',
                    'Interest income,Bank,15.25,Income,Interest,0',
                    'Dividend payment,Investment Firm,75.50,Income,Dividends,0',
                    'Car repair,Auto Shop,350.00,Transportation,Maintenance,1'
                ]

                # Write the sample data to the file
                with open(data_path, 'w') as f:
                    for line in sample_data:
                        f.write(line + '\n')

                logger.info(f"Created sample dataset with {len(sample_data)-1} transactions")

            # Run the command
            training_status['progress'] = 20
            training_status['message'] = f'Running command: {" ".join(cmd)}'
            logger.info(f"Running command: {' '.join(cmd)}")

            # Create models directory if it doesn't exist
            models_dir = os.path.join(ml_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)

            # Set up environment variables to activate the virtual environment
            env = os.environ.copy()
            venv_dir = os.path.dirname(os.path.dirname(venv_python))

            # Add the virtual environment's site-packages to PYTHONPATH
            site_packages = os.path.join(venv_dir, 'Lib', 'site-packages')
            if os.path.exists(site_packages):
                logger.info(f"Adding site-packages to PYTHONPATH: {site_packages}")
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{site_packages}{os.pathsep}{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = site_packages

            # Run the process with a timeout to prevent hanging
            try:
                logger.info(f"Running command with environment: PYTHONPATH={env.get('PYTHONPATH', 'Not set')}")

                logger.info("Attempting to run command with timeout...")
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60,  # 60 second timeout
                        env=env
                    )

                    # Process the result
                    if result.returncode == 0:
                        logger.info("Command completed successfully")
                        logger.info(f"STDOUT: {result.stdout}")
                        training_status['progress'] = 90
                        training_status['message'] = 'Model training completed successfully!'

                        # Check if model files were created
                        performance_path = os.path.join(models_dir, 'performance.pkl')
                        metadata_path = os.path.join(models_dir, 'metadata.pkl')

                        if os.path.exists(performance_path) and os.path.exists(metadata_path):
                            logger.info(f"Model files created: {performance_path}, {metadata_path}")
                        else:
                            logger.error(f"Model files not created. Performance: {os.path.exists(performance_path)}, Metadata: {os.path.exists(metadata_path)}")
                            training_status['message'] = 'Training completed but model files were not created.'
                    else:
                        logger.error(f"Command failed with return code {result.returncode}")
                        logger.error(f"STDERR: {result.stderr}")
                        training_status['progress'] = -1
                        training_status['message'] = f'Error training model: {result.stderr}'

                except subprocess.TimeoutExpired:
                    logger.error("Command timed out after 60 seconds")
                    training_status['progress'] = -1
                    training_status['message'] = 'Training process timed out after 60 seconds'

            except Exception as e:
                training_status['progress'] = -1
                training_status['message'] = f'Error running training process: {str(e)}'
                logger.error(f"Error running training process: {str(e)}")

            # Change back to the original directory
            os.chdir(original_dir)

            # Complete the training
            training_status['progress'] = 100
            training_status['message'] = 'Training completed successfully!'
            training_status['end_time'] = time.time()

            # Calculate training time
            training_time = training_status['end_time'] - training_status['start_time']
            logger.info(f"Model training completed in {training_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        training_status['message'] = f"Error training model: {str(e)}"
        training_status['progress'] = -1  # Indicate error
    finally:
        training_status['is_training'] = False

@require_http_methods(["GET"])
def model_metrics_view(request):
    """View for displaying model metrics"""
    # This view is accessible to anyone
    metrics = get_model_metrics()

    # Add training status to the context
    context = {
        'metrics': metrics,
        'training_status': training_status
    }

    return render(request, 'expenses/model_metrics.html', context)

@require_http_methods(["GET"])
def model_metrics_api(request):
    """API endpoint for model metrics data"""
    # This API endpoint is accessible to anyone
    metrics = get_model_metrics()
    return JsonResponse(metrics)

@csrf_exempt
@require_http_methods(["POST"])
def train_model_view(request):
    """View for training the model"""
    global training_status

    # Check if training is already in progress
    if training_status['is_training']:
        return JsonResponse({
            'status': 'error',
            'message': 'Training is already in progress',
            'progress': training_status['progress'],
            'training_message': training_status['message']
        })

    # Get training parameters
    use_enhanced = request.POST.get('use_enhanced', 'true').lower() == 'true'
    use_sample = request.POST.get('use_sample', 'false').lower() == 'true'
    sample_size = int(request.POST.get('sample_size', '100')) if use_sample else None

    # Start training in a background thread
    training_thread = threading.Thread(
        target=train_model_task,
        args=(use_enhanced, sample_size)
    )
    training_thread.daemon = True
    training_thread.start()

    # Return response
    return JsonResponse({
        'status': 'success',
        'message': 'Model training started',
        'progress': 0,
        'training_message': 'Initializing training...'
    })

@require_http_methods(["GET"])
def training_status_api(request):
    """API endpoint for getting the current training status"""
    global training_status

    return JsonResponse({
        'is_training': training_status['is_training'],
        'progress': training_status['progress'],
        'message': training_status['message'],
        'start_time': training_status['start_time'],
        'end_time': training_status['end_time']
    })
