import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Loads data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses data by scaling features and encoding categorical variables."""
    try:
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Congestion', axis=1))
        
        # Encode categorical variables
        y = pd.get_dummies(data['Congestion']).values
        
        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForest classifier on the preprocessed data."""
    try:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
