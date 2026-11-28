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
    try:
        logging.info("Loading data from %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", str(e))
        raise

def preprocess_data(data):
    try:
        logging.info("Preprocessing data")
        # Assume 'IsHoliday' is a boolean column
        data['IsHoliday'] = data['IsHoliday'].astype(bool)
        # Convert 'weather_main' to numerical values
        weather_map = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3}
        data['weather_main'] = data['weather_main'].map(weather_map)
        # Drop unnecessary columns
        data = data.drop(['Unnamed: 0'], axis=1)
        # Split features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X, y):
    try:
        logging.info("Training model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", str(e))
        raise
