# pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, log_param, start_run, pyfunc, log_params

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    X = data.drop('Congestion', axis=1)
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
        model = train_model(X_train, y_train)
        pyfunc.log_model(model, 'model', registered_model_name='congestion_detector')
        log_params({'temp_min': np.min(X['temp']), 'temp_max': np.max(X['temp'])})
