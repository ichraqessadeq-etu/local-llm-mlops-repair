import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses data by scaling features and encoding target variable"""
    try:
        # Split data into features (X) and target (y)
        X = data.drop('Class', axis=1)
        y = data['Class']

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # One-hot encode target variable
        y_encoded = pd.get_dummies(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model"""
    try:
        # Train model using RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model using MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
