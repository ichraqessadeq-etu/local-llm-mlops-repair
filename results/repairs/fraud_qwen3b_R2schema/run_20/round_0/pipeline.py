import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or has no data.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'Class' to binary
        data['Class'] = data['Class'].map({0: 0, 1: 1})
        
        # Split features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Preprocessing failed: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        rf_model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        
        return rf_model
    except Exception as e:
        raise ValueError(f"Model training failed: {e}")
