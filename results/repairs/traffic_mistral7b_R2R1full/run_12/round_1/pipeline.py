import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, end_run, register_model, log_metric
import pytest

def load_data(file_path):
    data = pd.read_csv(file_path)
    data.set_index('Hour', inplace=True)
    return data

def preprocess_data(data):
    X = data[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'IsHoliday', 'DayOfWeek', 'Month']]
    y = data['Congestion']
    return (X, y)

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    with start_run():
        model = train_model(X_train=X, y_train=y)
        log_artifact(model, 'model')
        log_metric('train_loss', model.loss_)
        log_param('num_trees', model.n_estimators)
        register_model(model, 'random_forest_congestion_detector')
