import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log metrics
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        classification_rep = classification_report(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(classification_rep, "classification_report.txt")
        
        return model

def predict(model, X):
    """Predict using the trained model."""
    return model.predict(X)

def save_model(model, model_path):
    """Save the model to a specified path."""
    mlflow.pyfunc.save_model(path=model_path, python_model=model)

def load_model(model_path):
    """Load the model from a specified path."""
    return mlflow.pyfunc.load_model(model_path)

