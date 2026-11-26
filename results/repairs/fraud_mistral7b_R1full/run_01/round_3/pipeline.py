import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary data preprocessing like handling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming that no preprocessing is required.
    return data, data['Amount']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        pyfunc.log_model(model, "model", registered_model_name="random_forest")
        print(f"Model with version {mlflow.get_run_id()} has been logged.")
