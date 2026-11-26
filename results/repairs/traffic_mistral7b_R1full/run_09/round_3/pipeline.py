import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    data['clouds_all'] = data['clouds_all'].astype(int)  # Convert 'clouds_all' to integer
    X = data.drop('Congestion', axis=1)
    y = data['Congestion']
    return X, y

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact(model, "models")
        mlflow.log_metric("train_loss", model.fit_params['loss'])

    return model
