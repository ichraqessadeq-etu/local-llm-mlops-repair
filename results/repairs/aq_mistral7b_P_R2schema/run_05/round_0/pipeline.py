import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(X_train, y_train):
    with pyfunc.start_run(run_name="Anomaly_Detector"):
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        artifact_path = f"models/anomaly_detector_{track.active_run().info.run_id}"
        log_artifact(artifact="model", local_path=artifact_path, dataframe=model.feature_importances_)
        return model
