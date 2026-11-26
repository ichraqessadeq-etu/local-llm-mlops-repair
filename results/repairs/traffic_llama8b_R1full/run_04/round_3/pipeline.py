# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['Holiday'] = data['Holiday'].map({0: 0, 1: 1})

        # One-hot encoding for categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'Holiday'])

        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

        # Split data into features (X) and target (y)
        X = data.drop(['Congestion', 'weather_main_Clear', 'weather_main_Clouds', 'weather_main_Rain', 'weather_main_Snow', 'Holiday_0', 'Holiday_1'], axis=1)
        y = data['Congestion']

        return X, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise
