# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, end_run, register_model
from mlflow.sklearn import save_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month', 'Congestion']]
    return (data.drop('Congestion', axis=1), data['Congestion'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
