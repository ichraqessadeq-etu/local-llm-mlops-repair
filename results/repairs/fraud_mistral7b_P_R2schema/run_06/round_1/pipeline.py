import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data.drop('Amount', axis=1), data['Amount'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    with mlflow.start_run(run_name='Fraud Detection'):
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, 'model')
        mlflow.log_artifact(artifact_path='model', artifact_rel_path='models')
        print(f'Model training completed with tracking run id: {tracking_run_id}')
    return model