import pandas as pd
import random_forest as rf
from sklearn.model_selection import train_test_split
from mlflow import log_artifact, start_run, log_param, log_metric, end_run
import numpy as np

np.random.seed(42)  # Seed the random number generator

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary preprocessing like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming no missing values and no encoding is needed.
    return data[Features], data['Congestion']

def train_model(X_train, y_train):
    with start_run():
        model = rf.RandomForestClassifier(random_state=42)  # Set random_state for reproducibility
        model.fit(X_train, y_train)
        log_artifact(model, "model")
        log_metric("train_score", model.score(X_train, y_train))
        log_metric("test_score", model.score(X_test, y_test))
