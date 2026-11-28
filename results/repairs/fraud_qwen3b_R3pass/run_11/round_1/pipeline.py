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
        raise ValueError("The file does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        return X, y
    except KeyError:
        raise ValueError("The 'Class' column is missing in the dataset.")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"An error occurred while training the model: {e}")

def save_model(model, model_name):
    """Save the trained model using MLflow."""
    try:
        mlflow.set_experiment("fraud_detection")
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_name)
    except Exception as e:
        raise ValueError(f"An error occurred while saving the model: {e}")
