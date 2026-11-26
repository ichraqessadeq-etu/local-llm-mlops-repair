# pipeline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, end_run, register_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data['Features'], data['Congestion'])

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    with start_run():
        model = train_model(X, y)
        log_param('num_trees', model.n_estimators)
        log_artifact("model", model)
        log_metric('train_loss', model.fit(X, y).score(X, y))
