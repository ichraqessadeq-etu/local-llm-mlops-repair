import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed
    X = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_Main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']]
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "random-forest")

    return model
