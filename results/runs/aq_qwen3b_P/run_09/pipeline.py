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
    """Preprocess the data by splitting features and target."""
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"An error occurred while training the model: {e}")

def predict_anomaly(model, X_test):
    """Predict anomalies using the trained model."""
    try:
        predictions = model.predict(X_test)
        return predictions
    except Exception as e:
        raise ValueError(f"An error occurred while predicting anomalies: {e}")
