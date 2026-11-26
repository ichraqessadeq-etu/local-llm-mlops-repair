import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise
    except pd.errors.ParserError:
        logger.error("Error parsing the CSV file.")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Convert 'Class' to binary for simplicity
        data['Class'] = data['Class'].map({0: 0, 1: 1})
        
        # Define features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Data preprocessed successfully.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Print classification report
        print(classification_report(y_test, y_pred))
        
        logger.info("Model evaluated successfully.")
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def main():
    # Load data
    data = load_data('data/fraud.csv')
    
    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

