import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data by splitting into features and target."""
    X = data.drop(columns=['Anomaly'])
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log it using MLflow."""
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("features", list(X_train.columns))
        mlflow.log_param("target", "Anomaly")
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(model, "model")
        
        return model
