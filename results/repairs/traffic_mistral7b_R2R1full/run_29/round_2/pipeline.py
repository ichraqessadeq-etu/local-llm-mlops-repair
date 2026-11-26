import pandas as pd
import mlflow
from models.random_forest_classifier import RandomForestClassifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, scaling features, etc.
    # For simplicity, I'm assuming that the data is already preprocessed.
    return data['Features'], data['Congestion']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier.train_model(X_train, y_train)
        mlflow.log_artifact(model, "model")
