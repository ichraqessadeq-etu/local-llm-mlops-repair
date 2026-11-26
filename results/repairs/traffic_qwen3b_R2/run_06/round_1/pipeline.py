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
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        features = data.drop('Congestion', axis=1)
        target = data['Congestion']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to train model: {e}")
        raise

def main():
    file_path = 'data/traffic.csv'
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        
        # Log model to MLflow
        mlflow.set_experiment("traffic-congestion")
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
            mlflow.sklearn.log_model(model, "random_forest_model")
        
        logging.info("Pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
