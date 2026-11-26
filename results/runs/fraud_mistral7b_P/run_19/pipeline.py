import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data.drop('Class', axis=1), data['Class']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    track(f"model.params", model.get_params())
    return model

def save_model(model, artifact_path):
    pyfunc.save_model(model, artifact_path)
    log_artifact(artifact_path)
