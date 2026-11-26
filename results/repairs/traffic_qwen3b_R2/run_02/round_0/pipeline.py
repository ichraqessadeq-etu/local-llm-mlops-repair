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
        # Drop columns with missing values
        data.dropna(inplace=True)
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].astype('category').cat.codes
        
        # Define features and target
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Data preprocessed successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        with mlflow.start_run():
            # Create and train the model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "congestion_detector")
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model accuracy: {accuracy}")
            
            return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
