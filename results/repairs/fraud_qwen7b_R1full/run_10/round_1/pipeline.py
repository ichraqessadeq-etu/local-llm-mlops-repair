# pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
            return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")
