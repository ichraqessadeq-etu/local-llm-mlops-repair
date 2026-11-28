import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Anomaly'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        mlflow_run_id = mlflow.active_run().info.run_id
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, artifact_path='model', registered_model_name='AnomalyDetector')
        log_artifact(artifact='model.pkl', local_path='model')
        print(f'Model saved with run id: {mlflow_run_id}')
        return model