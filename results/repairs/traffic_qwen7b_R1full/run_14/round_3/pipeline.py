import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.exceptions import NotFittedError

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data to prepare for modeling."""
    # Assuming 'Congestion' is the target column
    X = data.drop(columns=['Congestion'])
    y = data['Congestion']
    
    # Clean the data by converting non-numeric values to NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", X_train.columns.tolist())
        model = RandomForestClassifier(random_state=42)  # Add random_state for reproducibility
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", model.score(X_train, y_train))
    return model

def predict(model, X):
    """Predict congestion using a trained model."""
    try:
        return model.predict(X)
    except NotFittedError:
        raise ValueError("Model is not fitted. Please train the model first.")
