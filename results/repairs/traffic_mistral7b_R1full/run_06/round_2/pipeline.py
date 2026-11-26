import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

def load_data(file_path):
    pd.options.mode.use_inf_as_na = True  # Handle inf/-inf as NaN
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, scaling features, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    data = pd.get_dummies(data)  # One-hot encoding for categorical variables
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    np.random.seed(42)  # Set a global seed for reproducibility
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', local_model=model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit_score(X_train, y_train))
    return model
