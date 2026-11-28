import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Load air quality data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess data by selecting features and encoding categorical variables"""
    try:
        # Select features
        X = data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']]

        # Encode categorical variables
        X['Hour'] = pd.Categorical(X['Hour']).codes
        X['DayOfWeek'] = pd.Categorical(X['DayOfWeek']).codes
        X['Month'] = pd.Categorical(X['Month']).codes

        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split data into features and target
        y = data['Anomaly']

        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier on the preprocessed data"""
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
