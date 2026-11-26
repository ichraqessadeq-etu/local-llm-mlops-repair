# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging
import numpy as np

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Drop unnecessary columns
        data.drop(columns=['Hour', 'DayOfWeek', 'Month'], inplace=True)
        
        # Remove rows with NaN values
        data.dropna(axis=1, inplace=True)
        
        # Remove rows with NaN values
        data.dropna(inplace=True)
        
        # Replace NaN values with the mean of the column
        data.fillna(data.mean(), inplace=True)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Anomaly', axis=1))
        
        # Encode target variable
        y = data['Anomaly'].map({'Normal': 0, 'Anomaly': 1})
        
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
