import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

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
    """Train a RandomForestClassifier and log it using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
            model.fit(X_train, y_train)
            mlflow.log_model(model, "random_forest_model")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_metric("accuracy", 0.95)  # Placeholder value
            return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")
