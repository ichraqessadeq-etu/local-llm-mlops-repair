import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc
import mlflow.tracking

def log_model(model, artifact_path):
    with mlflow.start_run():
        mlflow.pyfunc.log_model(model, artifact_path)

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['Features'].values, data['Congestion'].values

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    artifact_path = "model"
    mlflow.tracking.MlflowClient().log_model(model, artifact_path)
