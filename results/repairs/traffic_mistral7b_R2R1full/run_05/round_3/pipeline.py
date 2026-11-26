# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data.replace({'weather_Main': {'Rain': 0, 'Snow': 0, 'Clouds': 1, 'Clear': 1}}, inplace=True)  # Replacing non-numeric values with 0 or 1
    data.dropna(inplace=True)  # Dropping NaN values
    return (data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_Main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']], data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
    return model
