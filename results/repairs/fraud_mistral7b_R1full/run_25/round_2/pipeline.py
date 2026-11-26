import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any preprocessing steps here, such as normalization, encoding, etc.
    # For simplicity, I'll assume no preprocessing is needed.
    return data['Features'], data['Class']

def train_model(X_train, y_train, random_state=42):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train, y_train)
        mlflow.log_artifact(f'model', model)
        mlflow.log_param('num_trees', model.n_estimators)
        mlflow.log_metric('train_loss', model.fit(X_train, y_train).score(X_train, y_train))
        mlflow.log_metric('val_loss', model.fit(X_train, y_train).score(X_val, y_val))

    return model
