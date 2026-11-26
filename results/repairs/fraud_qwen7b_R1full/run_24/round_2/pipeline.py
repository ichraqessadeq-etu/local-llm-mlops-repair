import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    try:
        X = data.drop(columns=['Class'])
        y = data['Class']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            mlflow.log_param("features", X_train.columns.tolist())
            mlflow.log_param("target", "Class")
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(model, "model")
            
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
