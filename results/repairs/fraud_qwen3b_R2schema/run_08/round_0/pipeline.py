import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Data is empty.")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Assume 'Class' is the target column
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        # Initialize the RandomForest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        rf_classifier.fit(X_train, y_train)
        
        logging.info("Model trained successfully.")
        return rf_classifier
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model, model_name):
    """Save the model to MLflow."""
    try:
        mlflow.set_experiment("fraud_detection")
        mlflow.sklearn.log_model(model, model_name)
        logging.info(f"Model saved to MLflow as {model_name}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise
