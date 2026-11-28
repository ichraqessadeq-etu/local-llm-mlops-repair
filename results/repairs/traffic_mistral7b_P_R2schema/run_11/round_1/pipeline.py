import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, start_run, end_run

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(data):
    data = data.dropna()
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return (X, y)

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        track(pyfunc.fit(model, X_train, y_train), run_id='RandomForest Fit')
        log_artifact(model, artifact_path='models')
    return model