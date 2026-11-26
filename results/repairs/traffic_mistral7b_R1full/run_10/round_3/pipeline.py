import pandas as pd
import mlflow
import randomforest_opus as rf
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any preprocessing steps here, such as normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data, data['Congestion']

def train_model(X_train, y_train):
    np.random.seed(42)  # Seed random state for reproducibility
    with mlflow.start_run():
        model = rf.RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(model, "models")
        mlflow.log_metric("train_score", model.score(X_train, y_train))
        mlflow.log_metric("test_score", model.score(X_test, y_test))  # Assuming you have X_test and y_test prepared

    return model
