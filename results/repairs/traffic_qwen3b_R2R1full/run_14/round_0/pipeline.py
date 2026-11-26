import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty file: %s", file_path)
        raise

def preprocess_data(data):
    """Preprocess the data."""
    logging.info("Preprocessing data")
    # Example preprocessing: drop columns with missing values
    data.dropna(inplace=True)
    # Example preprocessing: one-hot encode categorical variables
    data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday'])
    # Split features and target
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    logging.info("Training model")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logging.info("Accuracy: %.2f", accuracy)
    logging.info("Classification Report:\n%s", report)
    return accuracy, report

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)
        logging.info("Pipeline completed successfully")
        return accuracy, report
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    run_pipeline('data/traffic.csv')
