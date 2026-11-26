import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import logging

# Initialize logging
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
        # Convert target column to binary
        data['Anomaly'] = data['Anomaly'].map({'Anomaly': 1, 'Normal': 0})
        
        # Define features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate evaluation metrics
        report = classification_report(y_test, y_pred)
        confusion_matrix = confusion_matrix(y_test, y_pred)
        
        logging.info("Model evaluated successfully.")
        return report, confusion_matrix
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_model(model):
    """Save the model to MLflow."""
    try:
        mlflow.set_experiment("AirQualityAnomalyDetection")
        mlflow.sklearn.log_model(model, "anomaly_detector_model")
        logging.info("Model saved to MLflow successfully.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise
