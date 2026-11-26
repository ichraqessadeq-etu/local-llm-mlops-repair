import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here, such as normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data, data['Amount']

def train_test_split_consistent_length(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if len(X_train) != len(X_test):
        X_train, X_test = X_test, X_train
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(pyfunc.DistributedModel.from_flask(model))
        mlflow.log_artifact("model", local_path=".")

    return model
