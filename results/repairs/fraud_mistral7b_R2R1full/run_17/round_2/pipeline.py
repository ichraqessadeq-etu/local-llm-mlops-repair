import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, pyfunc, end_run
from sklearn.utils.multiclass import type_of_target
import yaml

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return (data, data['Amount'].astype('int'))

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, 'random_forest')
        model_yaml = yaml.safe_dump(model)
        log_artifact(model_yaml, 'model.yaml')
    return model
