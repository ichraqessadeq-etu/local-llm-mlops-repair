import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data, data['Anomaly']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        pyfunc.log_model(model, artifact_path="model")
        track(f"train_score", model.score(X_train, y_train))
        track(f"test_score", model.score(X_test, y_test))

X, y = preprocess_data(load_data('data/air_quality.csv'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_model(X_train, y_train)
