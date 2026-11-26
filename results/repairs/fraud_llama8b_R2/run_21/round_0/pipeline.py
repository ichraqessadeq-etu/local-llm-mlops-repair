import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Split data into features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("Data preprocessed successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
