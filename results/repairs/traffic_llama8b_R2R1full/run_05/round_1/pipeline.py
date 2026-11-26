# pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Handle missing values
        data = data.dropna()
        
        # Convert categorical variables to numerical
        data['weather_main'] = data['weather_main'].map({'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Snow': 3})
        data['Month'] = data['Month'].map({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11})
        
        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
