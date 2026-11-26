import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()
    return data.iloc[:, :-1], data.iloc[:, -1]

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="congestion_detector")
        track(f"Congestion Detector Training Time", model.best_estimator_.get_params()["n_estimators"])

    return model
