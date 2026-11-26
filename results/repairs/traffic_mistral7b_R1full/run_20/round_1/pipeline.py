import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Define the Features list
Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming no preprocessing is needed
    return data[Features], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)  # Seed the random number generator
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
