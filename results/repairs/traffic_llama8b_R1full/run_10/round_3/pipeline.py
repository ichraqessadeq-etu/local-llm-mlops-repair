# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn
import mlflow
import numpy as np

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
        data['weather_main'] = data['weather_main'].map({'Rain': 1, 'Snow': 2, 'Clear': 3, 'Clouds': 4, 'Thunderstorm': 5})
        
        # One-hot encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'weather_main_1', 'weather_main_2', 'weather_main_3', 'weather_main_4', 'weather_main_5', 'Holiday'])
        
        # Scale numerical features
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        # Handle missing values
        X = X.fillna(0)
        y = y.fillna(0)
        
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Import RandomForestClassifier class
        from sklearn.ensemble import RandomForestClassifier
        
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
