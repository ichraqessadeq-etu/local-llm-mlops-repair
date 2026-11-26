import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    logging.info("Loading data from %s", file_path)
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise

def preprocess_data(data):
    """Preprocess data by splitting features and target."""
    logging.info("Preprocessing data")
    try:
        # Convert categorical columns to numerical using one-hot encoding
        data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday', 'DayOfWeek', 'Month'])
        
        # Convert 'Congestion' to binary for classification
        data['Congestion'] = data['Congestion'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        features = data.drop(columns=['Congestion'])
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error preprocessing data: %s", e)
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    logging.info("Training model")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", e)
        raise

def save_model(model, file_path):
    """Save the trained model."""
    logging.info("Saving model to %s", file_path)
    try:
        mlflow.sklearn.save_model(model, file_path)
        logging.info("Model saved successfully")
    except Exception as e:
        logging.error("Error saving model: %s", e)
        raise

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model(model, 'models/random_forest_congestion_detector.pkl')
        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        raise
