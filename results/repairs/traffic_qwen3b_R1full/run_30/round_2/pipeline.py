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
        logging.info("Loading data from %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data in file: %s", file_path)
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        logging.info("Preprocessing data")
        # Convert target column to binary
        data['Congestion'] = data['Congestion'].map({'high': 1, 'low': 0})
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Convert 'weather_main' to a categorical variable
        X['weather_main'] = X['weather_main'].astype('category')
        
        # Convert all columns to numeric, coercing non-numeric values to NaN
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Handle NaN values by dropping rows with any NaN values
        X.dropna(inplace=True)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        logging.info("Training model")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "congestion_detector")
        
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        raise

def main():
    file_path = 'data/traffic.csv'
    try:
        data = load_data(file_path)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Model accuracy: %.2f", accuracy)
    except Exception as e:
        logging.error("Pipeline failed: %s", str(e))

if __name__ == "__main__":
    main()
