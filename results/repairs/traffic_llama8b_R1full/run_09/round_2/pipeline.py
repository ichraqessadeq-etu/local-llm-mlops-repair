# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    try:
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Congestion', axis=1))

        # Split data into features (X) and target (y)
        y = data['Congestion']
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise
