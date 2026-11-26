# pipeline.py
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow.pyfunc import log_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'].values, data['Congestion'].values)

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, 'model')
        mlflow.log_metric('train_score', model.score(X_train, y_train))
