import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'])

def train_model(X_train, y_train):
    with pyfunc.start_run(run_name='RandomForest_Fraud_Detector'):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_artifact(artifact=f'model_{track.active_run().info.run_id}.pkl', artifact_location='models')
    return model