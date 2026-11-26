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
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path, dtype={
            'temp': float,
            'rain_1h': float,
            'snow_1h': float,
            'clouds_all': float,
            'Hour': int,
            'DayOfWeek': int,
            'Month': int
        })
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Empty file.")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file.")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting features and target."""
    try:
        features = data.drop('Congestion', axis=1)
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        logger.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        return accuracy
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise
