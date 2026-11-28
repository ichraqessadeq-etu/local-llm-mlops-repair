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
    """Load data from the specified CSV file."""
    try:
        logging.info("Loading data from file: %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error("Error loading data: %s", str(e))
        raise

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        logging.info("Preprocessing data")
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
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
        logging.info("Model accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", str(e))
        raise

def main(file_path):
    """Main function to execute the pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        logging.info("Pipeline completed successfully")
    except Exception as e:
        logging.error("Pipeline failed: %s", str(e))

if __name__ == "__main__":
    main('data/traffic.csv')
