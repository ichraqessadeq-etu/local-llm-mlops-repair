import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def load_data(file_path):
    """Loads data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocesses data by encoding categorical variables and scaling"""
    try:
        # Encode categorical variables
        data['weather_main'] = pd.Categorical(data['weather_main']).codes
        data['Month'] = pd.Categorical(data['Month']).codes
        data['DayOfWeek'] = pd.Categorical(data['DayOfWeek']).codes
        
        # Scale numerical variables
        scaler = StandardScaler()
        data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour']] = scaler.fit_transform(data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour']])
        
        # Split data into features (X) and target (y)
        X = data.drop('Congestion', axis=1)
        y = data['Congestion']
        
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Trains a RandomForest model"""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random_forest")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        raise
