import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        logging.info("Loading data from %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data in file: %s", file_path)
        raise

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target, and handling missing values."""
    try:
        logging.info("Preprocessing data")
        features = data.drop('Congestion', axis=1)
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    try:
        logging.info("Training model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", str(e))
        raise
