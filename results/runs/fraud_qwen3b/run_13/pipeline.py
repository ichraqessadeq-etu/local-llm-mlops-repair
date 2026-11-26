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
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Assuming 'Class' is the target column
        X = data.drop(columns=['Class'])
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Model accuracy: {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def main(file_path):
    """Main function to execute the pipeline."""
    try:
        # Load data
        data = load_data(file_path)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Log model to MLflow
        mlflow.log_model(model, "model")
        logging.info("Model logged to MLflow.")
    except Exception as e:
        logging.error(f"Error in pipeline execution: {e}")

if __name__ == "__main__":
    main("data/fraud.csv")
