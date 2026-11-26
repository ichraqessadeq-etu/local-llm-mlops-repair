import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import PyFuncModel, track
from mlflow.tracking import MlflowClient

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'll assume no preprocessing is required.
    return data['Features'], data['Class']

def train_model(X_train, y_train):
    with MlflowClient() as client:
        experiment_id = client.create_experiment("Fraud_Detection_Experiment")
        run_id = client.create_run(experiment_id)
        client.log_param("random_state", 42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        artifact_path = f"models/random_forest_{str(int(time.time()))}"
        client.log_artifact(artifact_path, "model")

        mlflow_model = PyFuncModel.from_sklearn(model, "predict")
        mlflow_model.save(artifact_path)
        client.log_model(mlflow_model, "model", artifact_path)
