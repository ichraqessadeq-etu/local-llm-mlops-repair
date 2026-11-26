import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    data = data.dropna()  # Drop missing values
    X = data.drop('Congestion', axis=1)  # Features
    y = data['Congestion']  # Target
    return X, y

def train_model(X_train, y_train, random_state=42):
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train(file_path, random_state=42):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    with mlflow.start_run():
        model = train_model(X_train, y_train, random_state=random_state)
        mlflow.pyfunc.log_model(model, "model", registered_model_name="congestion_detector")
