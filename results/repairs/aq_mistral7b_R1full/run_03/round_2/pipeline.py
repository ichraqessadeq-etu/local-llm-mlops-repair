import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, run_local
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Any preprocessing steps like normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        mlflow.log_metric("train_score", model.score(X_train, y_train))

    return model
