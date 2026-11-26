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
    """Load data from a CSV file."""
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
    except pd.errors.ParserError:
        logging.error("Error parsing CSV file.")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['IsHoliday'] = data['IsHoliday'].astype(int)
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        with mlflow.start_run():
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Log model
            mlflow.sklearn.log_model(rf, "random_forest_model")
            
            # Evaluate model
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy}")
            
            return rf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
