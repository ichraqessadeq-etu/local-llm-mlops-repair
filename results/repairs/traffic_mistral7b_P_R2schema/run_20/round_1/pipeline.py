import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return (np.array(data.drop('Congestion', axis=1)), data['Congestion'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    with mlflow.start_run():
        model.fit(X_train, y_train)
        pyfunc.log_model(pyfunc.PyFuncModel.from_sklearn(model), artifact_path='model')
        track(f'Congestion_model_version', model.version)
    return model