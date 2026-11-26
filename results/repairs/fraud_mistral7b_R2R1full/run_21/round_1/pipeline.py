import pandas as pd
import mlflow
import random_forest_classifier

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform necessary data preprocessing such as scaling, encoding, handling missing values, etc.
    # Return X (features) and y (target)

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = random_forest_classifier.train(X_train, y_train)
        mlflow.log_artifact(model, "model")
