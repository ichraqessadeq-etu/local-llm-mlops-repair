import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import pyfunc, track, active_run

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']], data['Class'])

def train_model(X_train, y_train):
    with mlflow.start_run(run_name='RandomForest-Fraud-Detector'):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(pyfunc.DistributedLogModel.from_sklearn(model), 'model')
        track(active_run().info.run_id)
    return model