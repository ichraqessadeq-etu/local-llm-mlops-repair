import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty dataset")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and splitting into training and testing sets."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        return accuracy
    except Exception as e:
        raise ValueError(f"Error evaluating model: {e}")
