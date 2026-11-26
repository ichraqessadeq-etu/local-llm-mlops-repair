import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data, data['Amount']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        mlflow.log_artifact(artifact_path="model", artifact_rel_path="models")
        print(f"Model trained and logged with run_id: {mlflow.get_run_id()}")
        track(f"Model trained successfully")
