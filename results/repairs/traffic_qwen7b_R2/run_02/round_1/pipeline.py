import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric
from mlflow.sklearn import log_model

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data to split into features and target."""
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model parameters and metrics
    log_param("n_estimators", model.n_estimators)
    log_param("random_state", model.random_state)
    
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    log_metric("train_accuracy", accuracy)
    
    log_model(model, "random_forest_model")
    
    return model
