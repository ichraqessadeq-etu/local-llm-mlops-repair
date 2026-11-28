import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Congestion'])

def train_model(X_train, y_train):
    with mlflow.start_run(run_name='RandomForestCongestionDetector'):
        mlflow.log_artifact()
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, 'model')
        print(f'Model training completed with MLflow run ID: {mlflow.get_run_id()}')