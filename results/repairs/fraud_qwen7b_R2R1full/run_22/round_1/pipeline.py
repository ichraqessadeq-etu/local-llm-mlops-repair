import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by separating features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        y_pred = rf_model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Model accuracy: {accuracy}")
        print(classification_report(y_train, y_pred))
        return rf_model

