import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Loads the air quality data from the CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses the data by scaling and encoding categorical features."""
    try:
        # Scale numeric features
        scaler = StandardScaler()
        data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']] = scaler.fit_transform(data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']])

        # Encode categorical features
        data['Hour'] = pd.Categorical(data['Hour']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes

        # Split data into features (X) and target (y)
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']

        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a Random Forest classifier on the preprocessed data."""
    try:
        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "random_forest")

        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {e}")
        raise
