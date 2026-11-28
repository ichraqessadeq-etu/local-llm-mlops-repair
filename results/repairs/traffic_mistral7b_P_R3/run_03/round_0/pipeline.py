import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import tracking, pyfunc, sklearn

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return data, data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model = pyfunc.fit(model, X_train, y_train)
        mlflow.sklearn.log_model(model, "congestion_detector")
        tracking.log_param("n_estimators", model.n_estimators)
        tracking.log_param("max_depth", model.max_depth)
        tracking.log_param("min_samples_split", model.min_samples_split)
        tracking.log_param("min_samples_leaf", model.min_samples_leaf)
        return model
