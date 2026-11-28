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
    with mlflow.start_run():
        track('experiment_name', 'Fraud Detection')
        track('run_id', f'{str(hash(str(X_train)))[:8]}')
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(pyfunc.DistributedModel.from_flask(model), artifact_path='model')
        print(f'Model trained with accuracy: {model.score(X_train, y_train)}')