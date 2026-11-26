import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data.drop('Anomaly', axis=1), data['Anomaly'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    with mlflow.start_run():
        mlflow.log_artifact(f"models/model.pkl", model)
        mlflow.log_metric('train_score', model.score(X_train, y_train))
    return model

def save_model(model, model_path):
    joblib.dump(model, model_path)
