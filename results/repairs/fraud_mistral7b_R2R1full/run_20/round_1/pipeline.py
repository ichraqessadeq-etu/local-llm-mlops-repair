# pipeline.py
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    pass

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        # mlflow.tracking.log_param("learning_rate", 0.1) # Uncomment this line if you want to log a parameter
        return model
