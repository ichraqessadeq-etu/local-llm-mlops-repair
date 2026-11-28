import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    pass

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, 'model')
        pass  # R3: API inexistante retirée
        return model