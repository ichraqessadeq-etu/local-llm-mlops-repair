import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    """Preprocess data by scaling and encoding categorical variables"""
    try:
        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

        # Encode categorical variables
        data['weather_main'] = pd.Categorical(data['weather_main']).codes
        data['IsHoliday'] = pd.Categorical(data['IsHoliday']).codes

        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']

        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Train a RandomForest model on the preprocessed data"""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest_model")
        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {str(e)}")
        raise
