import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, start_run, end_run, pyfunc
from mlflow.tracking import Metric

def log_metrics(metrics):
    for name, value in metrics.items():
        Metric.log_metric(name, value)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add your preprocessing steps here
    return data[Features], data['Congestion']

def train_model(X_train, y_train, random_state=42):
    with start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        artifact_path = f"model"
        log_artifact(model, artifact_path)
        log_metrics({"train_score": model.score(X_train, y_train)})
    return model

def get_random_state():
    return pyfunc.get_artifact("random_state")

def set_random_state(random_state):
    pyfunc.log_artifact(random_state, "random_state")
