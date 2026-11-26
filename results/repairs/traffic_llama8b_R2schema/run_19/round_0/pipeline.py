import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        mlflow.log_error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data):
    try:
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # Convert categorical features
        data['weather_main'] = pd.Categorical(data['weather_main']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes

        # Scale features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

        # Split data into X and y
        X = data.drop(['Congestion', 'weather_main', 'DayOfWeek', 'Month'], axis=1)
        y = data['Congestion']

        return X, y
    except Exception as e:
        mlflow.log_error(f"Error preprocessing data: {str(e)}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        mlflow.log_error(f"Error training model: {str(e)}")
        raise
