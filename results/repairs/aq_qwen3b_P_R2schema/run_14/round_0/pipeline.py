import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Drop rows with missing values
        data = data.dropna()
        
        # Convert target column to binary
        data['Anomaly'] = data['Anomaly'].map({'Anomaly': 1, 'Normal': 0})
        
        # Define features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            
            # Train the model
            model = RandomForestClassifier(n_estimators=100, max_depth=10)
            model.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(model, "anomaly_detector_model")
            
            logging.info("Model trained successfully.")
            return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
