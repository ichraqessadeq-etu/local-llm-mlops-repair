# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Separate categorical and numerical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

        # Scale numerical columns
        scaler = StandardScaler()
        X = scaler.fit_transform(data[numerical_cols])

        # One-hot encode categorical columns
        X = pd.get_dummies(data[categorical_cols], drop_first=True)

        # Split data into features (X) and target (y)
        y = data['Congestion']
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
