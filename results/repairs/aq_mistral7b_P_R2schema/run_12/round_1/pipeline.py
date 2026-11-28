import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[features], data['Anomaly'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    track(f'Training RandomForestClassifier')
    model.fit(X_train, y_train)
    return model

def save_model(model, experiment_name):
    mlflow.set_tracking_artifact(model, artifact_path=f'models/{experiment_name}')