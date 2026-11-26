import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    return (data, data['Amount'].astype(np.float64))

def type_of_target(y):
    if y.dtype == np.float64:
        return "continuous"
    else:
        return super().type_of_target(y)

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'random-forest')
    return model
