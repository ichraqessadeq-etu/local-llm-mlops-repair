import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return np.array(data.drop('Congestion', axis=1)), data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="congestion_detector")
        track(f"Congestion Detector Training Loss", model.loss_)
        track(f"Congestion Detector Training Score", model.score(X_train, y_train))
        return model
