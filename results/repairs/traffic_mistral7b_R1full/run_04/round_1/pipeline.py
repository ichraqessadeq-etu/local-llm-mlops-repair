import pandas as pd
import mlflow
import randomforest as rf
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling missing values, encoding categorical variables, etc.
    # For simplicity, I'm assuming the data is clean and ready to use.
    return data, data['Congestion']

def train_model(X_train, y_train):
    np.random.seed(42)  # Seed the random number generator for reproducibility
    with mlflow.start_run():
        model = rf.RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(model, "model")
        mlflow.log_metric("train_loss", model.fit(X_train, y_train).loss_)

    return model
