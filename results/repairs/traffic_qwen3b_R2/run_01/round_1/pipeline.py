import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from the specified file path."""
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
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        logging.info("Data preprocessed successfully.")
        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model evaluation: Accuracy = {accuracy}, Report = {report}")
        return accuracy, report
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def main(file_path):
    """Main function to orchestrate the pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            accuracy, report = evaluate_model(model, X_test, y_test)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "congestion_detector")
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text("classification_report", report)
            
            logging.info("Pipeline completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main('data/traffic.csv')
