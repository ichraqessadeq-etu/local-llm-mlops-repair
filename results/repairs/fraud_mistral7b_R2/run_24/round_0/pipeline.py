import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, track

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps here (e.g., normalization, encoding, feature engineering)
    X = data.drop('Class', axis=1)
    y = data['Class']
    return X, y

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        mlflow.log_artifact("model.pkl", "model")
        track(f"Training loss: {model.loss_}")
        track(f"Training accuracy: {model.score(X_train, y_train)}")
