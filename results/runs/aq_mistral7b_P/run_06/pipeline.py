import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("data", local_path=".")
        pyfunc.log_model(model, artifact_path="model")
        track(f"Training Accuracy", model.score(X_train, y_train))
        track(f"Testing Accuracy", model.score(X_test, y_test))

X_train, y_train = preprocess_data(load_data('data/air_quality.csv'))
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
train_model(X_train, y_train)
