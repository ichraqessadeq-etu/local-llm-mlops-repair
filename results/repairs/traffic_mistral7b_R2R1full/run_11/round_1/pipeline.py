# pipeline.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, log_metric, pyfunc, start_run, end_run

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (np.array(data.drop('Congestion', axis=1)), data['Congestion'])

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, 'model')
        log_metric('train_score', model.score(X_train, y_train))
    return model
