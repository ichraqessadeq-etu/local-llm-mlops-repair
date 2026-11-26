import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming that the data is already preprocessed
    X = data[data.columns[:-1]]  # Assuming 'Congestion' is the target column
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact(model, "models")
        mlflow.log_metric("train_loss", model.fit_params['loss'])

    return model
