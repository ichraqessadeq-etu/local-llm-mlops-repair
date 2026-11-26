import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_model(model, "model", signature=infer_signature(X_train, y_train))
        mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        return model
