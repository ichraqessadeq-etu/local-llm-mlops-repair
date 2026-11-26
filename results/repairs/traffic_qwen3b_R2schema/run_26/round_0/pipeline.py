import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and splitting into training and testing sets."""
    try:
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier and log the model using MLflow."""
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        
        # Log the model using MLflow
        mlflow.sklearn.log_model(rf, "congestion_detector_model")
        logging.info("Model logged successfully.")
        
        return rf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
