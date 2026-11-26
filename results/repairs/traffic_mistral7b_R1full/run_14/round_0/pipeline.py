import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, end_run, register_model, save_model

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary preprocessing like filling NaN, encoding categorical variables, etc.
    # For simplicity, I'm assuming data is already preprocessed
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
