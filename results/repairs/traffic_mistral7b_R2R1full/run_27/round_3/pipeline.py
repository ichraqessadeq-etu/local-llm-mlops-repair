# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, start_run, end_run, log_metric

Features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'Hour', 'DayOfWeek', 'Month']

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data[Features] = data[Features].astype(float)
    return (data[Features], data['Congestion'])

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        artifact_path = f'model'
        log_artifact(model, artifact_path)
        log_metric('train_score', model.score(X_train, y_train))
    return model
