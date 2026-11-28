import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data.drop('Anomaly', axis=1), data['Anomaly'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, 'anomaly_detector')
        run_id = tracking.active_run().info.run_id
        log_artifact(f'anomaly_detector_{run_id}.pkl', 'anomaly_detector')
        return model