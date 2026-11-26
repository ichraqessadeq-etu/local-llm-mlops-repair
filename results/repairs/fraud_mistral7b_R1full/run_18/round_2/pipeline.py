import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, tracking

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here
    return data, data['Amount']

def train_model(X_train, y_train, random_state=42):
    with tracking.start_run():  # Replace 'mlflow.start_run()' with 'tracking.start_run()'
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="RandomForestFraudDetector")

    return model
