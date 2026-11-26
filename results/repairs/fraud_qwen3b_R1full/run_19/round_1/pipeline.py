import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data file.")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        logging.info("Data preprocessed successfully.")
        return X, y
    except KeyError:
        logging.error("Target column 'Class' not found.")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def predict(model, X_test):
    """Make predictions using the trained model."""
    try:
        y_pred = model.predict(X_test)
        logging.info("Predictions made successfully.")
        return y_pred
    except Exception as e:
        logging.error(f"Error making predictions: {e}")
        raise

def evaluate(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def run_pipeline(file_path):
    """Run the entire pipeline from loading data to evaluation."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        y_pred = predict(model, X_test)
        accuracy = evaluate(model, X_test, y_test)
        return accuracy
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

# Example usage
if __name__ == "__main__":
    file_path = 'data/fraud.csv'
    accuracy = run_pipeline(file_path)
    print(f"Model accuracy: {accuracy:.2f}")
