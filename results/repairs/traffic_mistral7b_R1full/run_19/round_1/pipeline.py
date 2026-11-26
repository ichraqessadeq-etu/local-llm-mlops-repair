import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import numpy as np

# Define Features
Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary data preprocessing like handling missing values, encoding categorical variables, etc.
    # For simplicity, let's assume no preprocessing is needed
    return data[Features], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random state for reproducibility
        model.fit(X_train, y_train)
        mlflow.log_artifact(artifact_path='model', local_model=model)
    return model
