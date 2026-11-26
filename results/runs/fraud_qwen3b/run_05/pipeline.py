import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_metric, log_param, log_artifact
import mlflow

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the dataset."""
    try:
        # Split the data into features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Preprocessing error: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics and parameters
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", model.n_estimators)
        
        return model
    except Exception as e:
        raise ValueError(f"Model training error: {e}")

def save_model(model, artifact_path):
    """Save the trained model."""
    try:
        model.save(artifact_path)
        log_artifact(artifact_path)
        print(f"Model saved to {artifact_path}")
    except Exception as e:
        raise ValueError(f"Model saving error: {e}")
