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
    logging.info("Loading data from %s", file_path)
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    """Preprocess the data."""
    logging.info("Preprocessing data")
    try:
        # Assume 'Anomaly' is the target column
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    logging.info("Training model")
    try:
        # Initialize the RandomForest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        rf.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        return rf
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def evaluate_model(rf, X_test, y_test):
    """Evaluate the model."""
    logging.info("Evaluating model")
    try:
        # Make predictions
        y_pred = rf.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %.2f", accuracy)
        
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", e)
        raise
