import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models import Model

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def preprocess_data(data):
    """
    Preprocess the data by splitting into features (X) and target (y).
    
    Args:
        data (pd.DataFrame): Data to preprocess.
    
    Returns:
        tuple: Features (X) and target (y).
    """
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
    
    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_model(model, "model", signature=infer_signature(X_train, y_pred))
            return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")
