import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, start_run, pyfunc, log_param

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    X = data.drop('Congestion', axis=1)  # Features
    y = data['Congestion']  # Target
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    with start_run():
        model = train_model(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="congestion_detector")
