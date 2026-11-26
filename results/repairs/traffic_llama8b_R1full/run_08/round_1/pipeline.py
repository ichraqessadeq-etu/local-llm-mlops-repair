# pipeline.py
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
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['Hour'] = pd.Categorical(data['Hour']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        data['Month'] = pd.Categorical(data['Month']).codes

        # One-hot encode categorical variables
        data = pd.get_dummies(data, columns=['Hour', 'DayOfWeek', 'Month'])

        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])

        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']

        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train a RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log the model with MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
