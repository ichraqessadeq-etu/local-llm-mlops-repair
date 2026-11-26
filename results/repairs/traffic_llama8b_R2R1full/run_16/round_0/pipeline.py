import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Loads the traffic data from the specified file path"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocesses the data by encoding categorical variables and scaling numerical variables"""
    try:
        # One-hot encode categorical variables
        data = pd.get_dummies(data, columns=['weather_main', 'IsHoliday', 'DayOfWeek', 'Month'])
        
        # Scale numerical variables
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None

def train_model(X_train, y_train):
    """Trains a RandomForest classifier on the preprocessed data"""
    try:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None
