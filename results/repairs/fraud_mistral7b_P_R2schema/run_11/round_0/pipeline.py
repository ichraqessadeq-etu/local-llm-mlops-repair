import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data.drop('Class', axis=1), data['Class']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(artifact=f"{model.feature_importances_}", artifact_path="feature_importances")
        pyfunc.log_model(model, artifact_path="pyfunc_model")
        track(f"Model Training Time", model.best_estimator_.n_iter_)

    return model
