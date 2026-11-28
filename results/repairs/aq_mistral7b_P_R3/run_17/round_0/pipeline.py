import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return data.iloc[:, :-1], data.iloc[:, -1]

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        mlflow.log_artifact(artifact_path="model.pkl", artifact_name="model")
        print(f"Model training completed with tracking run id: {mlflow.get_run_id()}")
