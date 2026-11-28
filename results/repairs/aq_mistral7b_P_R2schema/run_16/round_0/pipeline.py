import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data[Features], data['Anomaly']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    track(f"model.params", model.get_params())
    mlflow_model = pyfunc.PyFuncModel(train_instance=model.fit, predict_instance=model.predict)
    mlflow.sklearn.log_model(mlflow_model, "model")

