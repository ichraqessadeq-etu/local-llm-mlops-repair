import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting features and target."""
    try:
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def main():
    file_path = 'data/fraud.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
