import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it with MLflow."""
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_model(model, "model")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        mlflow.log_metric("accuracy", accuracy)
    return model
