import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")  # Use mlflow.sklearn.log_model
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            y_pred = model.predict(X_train)
            accuracy = accuracy_score(y_train, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"Model trained with accuracy: {accuracy}")
            print(classification_report(y_train, y_pred))
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")
