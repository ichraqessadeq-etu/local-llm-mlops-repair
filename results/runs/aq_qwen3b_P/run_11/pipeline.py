import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        features = data.drop('Anomaly', axis=1)
        target = data['Anomaly']
        logging.info("Data preprocessed successfully.")
        return features, target
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy}")
        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def main():
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    features, target = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
