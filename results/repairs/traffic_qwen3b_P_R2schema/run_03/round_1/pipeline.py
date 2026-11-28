import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Drop unnecessary columns
        data = data.drop(['Unnamed: 0'], axis=1)
        
        # Convert categorical columns to numerical
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        logging.info("Model trained successfully")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model, experiment_name, run_name):
    """Save the model using MLflow."""
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.sklearn.log_model(model, "model")
            logging.info("Model saved to MLflow")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise
