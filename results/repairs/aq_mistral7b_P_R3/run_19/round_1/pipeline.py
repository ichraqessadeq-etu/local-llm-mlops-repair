import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc
from mlflow.sklearn import save_model
from mlflow.sklearn import load_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (np.array(data.drop('Anomaly', axis=1)), data['Anomaly'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    pass  # R3: API inexistante retirée
    return model

def save_model_and_data(model, artifact_path):
    log_artifact(artifact_path + '/model', model)
    log_artifact(artifact_path + '/data', data)