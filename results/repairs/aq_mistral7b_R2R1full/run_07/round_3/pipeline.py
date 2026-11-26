import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from mlflow import log_param, log_artifact, start_run, end_run, log_metric

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Anomaly'])

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        log_param("random_state", model.get_params()["random_state"])
        pickle.dump(model, open("model.pkl", "wb"))
        log_artifact("model.pkl", "model")
        log_metric('train_score', model.score(X_train, y_train))
        log_metric('test_score', model.score(X_test, y_test))
