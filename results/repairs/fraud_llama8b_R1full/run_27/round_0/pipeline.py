import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """
    Load data from CSV file
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess data by scaling features and encoding target variable
    """
    try:
        # Scale features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Class', axis=1))
        
        # Encode target variable
        y = pd.get_dummies(data['Class']).iloc[:, 0]
        
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """
    Train RandomForest model
    """
    try:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
