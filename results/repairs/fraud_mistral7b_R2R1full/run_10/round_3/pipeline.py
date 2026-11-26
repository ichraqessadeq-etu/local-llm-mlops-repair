# pipeline.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import log_artifact, log_param

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Class'])

def train_model(X_train, y_train):
    with mlflow.start_run():
        log_param('random_state', 42)
        log_artifact(X_train, 'X_train')
        log_artifact(y_train, 'y_train')
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(pyfunc.DistributedModel.from_flask(model), artifact_path='model')
        print(f'Model with version {mlflow.get_run_id()} has been saved.')
