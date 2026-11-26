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
    """Preprocess data to split into features (X) and target (y)."""
    try:
        X = data.drop(columns=['Congestion'])
        y = data['Congestion']
        return X, y
    except Exception as e:
        raise Exception(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", model.score(X_train, y_train))
            return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")
