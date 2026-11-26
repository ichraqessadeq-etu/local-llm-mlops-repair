import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    logging.info("Loading data from %s", file_path)
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data."""
    logging.info("Preprocessing data")
    # Convert categorical variables to numerical
    data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
    data['IsHoliday'] = data['IsHoliday'].astype(int)
    # Split features and target
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    logging.info("Training model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Model accuracy: %.2f", accuracy)
    return accuracy

def run_pipeline(file_path):
    """Run the entire pipeline."""
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    return model
