import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data.drop(columns=['Class']), data['Class']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(artifact=model, artifact_path="models")
        pyfunc.log_model(model, artifact_path="models")
        track(f"Training Score: {model.score(X_train, y_train)}")
