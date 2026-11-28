import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data file.")
        raise

def preprocess_data(data):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    try:
        # Handle missing values
        data.fillna(method='ffill', inplace=True)
        
        # Encode categorical variables
        data['DayOfWeek'] = data['DayOfWeek'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
        
        # Split features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        logging.info("Data preprocessed successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Model training error: {e}")
        raise

def save_model(model, file_path):
    """Save the trained model to the specified file path."""
    try:
        mlflow.sklearn.save_model(model, file_path)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error(f"Model saving error: {e}")
        raise
