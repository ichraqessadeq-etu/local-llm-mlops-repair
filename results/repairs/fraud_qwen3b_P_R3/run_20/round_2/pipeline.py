import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

# Configure logging
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
    """Preprocess the dataset."""
    try:
        # Assume 'Class' is the target column
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data preprocessed successfully.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def log_model(model, experiment_name, run_name, artifact_path):
    """Log the trained model using MLflow."""
    try:
        # Start a new MLflow run
        with mlflow.start_run( run_name=run_name):
            # Log the model
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            logging.info("Model logged successfully.")
    except Exception as e:
        logging.error(f"Error logging model: {e}")
        raise

def main():
    # Define paths
    data_path = 'data/fraud.csv'
    experiment_name = 'fraud_detection'
    run_name = 'run_1'
    artifact_path = 'model'
    
    # Load data
    data = load_data(data_path)
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Log model
    log_model(model, experiment_name, run_name, artifact_path)

if __name__ == "__main__":
    main()
